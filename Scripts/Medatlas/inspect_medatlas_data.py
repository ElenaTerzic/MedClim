import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import re
import Basins

# -------- SETTINGS --------
data_dir = "/Users/elena/Projects/IRB/Manca_Argo/Data/Medatlas/NetCDF"  # adjust
files = {
    "Bottle": f"{data_dir}/Btl_data.nc",
    "CTD":    f"{data_dir}/CTD_data.nc",
    "XBT":    f"{data_dir}/XBT_data.nc",
    "MBT":    f"{data_dir}/MBT_data.nc",
}

basin = Basins.worldseas.med
USE_QC = False          # set True if you want to exclude "bad_quality"
GOOD_QC_ONLY = True     # if USE_QC=True: True keeps only QC=good (48); False keeps good+unknown+questionable
# -------------------------



def years_from_date_time(ds):
    """
    Convert ODV 'date_time' given as 'days since YYYY-MM-DD ... UTC' to calendar year,
    using the origin embedded in the NetCDF 'units' attribute.
    """
    days = ds["date_time"].values.astype(float)
    fill = ds["date_time"].attrs.get("_FillValue", None)

    bad = ~np.isfinite(days)
    if fill is not None:
        bad |= (days == fill)

    days = days.copy()
    days[bad] = np.nan

    units = ds["date_time"].attrs.get("units", "")
    m = re.search(r"days since (\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", units)
    if not m:
        # sometimes time part omitted
        m = re.search(r"days since (\d{4}-\d{2}-\d{2})", units)
        if not m:
            raise ValueError(f"Cannot parse date_time units: {units}")
        date0, time0 = m.group(1), "00:00:00"
    else:
        date0, time0 = m.group(1), m.group(2)

    t0 = pd.Timestamp(f"{date0}T{time0}Z")

    dt = t0 + pd.to_timedelta(days, unit="D")
    years = pd.DatetimeIndex(dt).year.to_numpy(dtype=float)
    years = np.where(np.isfinite(days), years, np.nan)

    return years, days, t0



def decode_odv_string(var):
    """
    Robust decoder for ODV NetCDF string-like variables.
    Accepts either an xarray DataArray or numpy array.

    Handles:
      A) (N, STRING) arrays of chars (bytes or unicode)
      B) (N,) arrays of fixed-length strings (dtype '|S20' or '<U20')
      C) integer arrays holding ASCII codes (uint8/int8), either (N, STRING) or (N,)
    Returns: np.ndarray of Python strings, shape (N,)
    """
    import numpy as np

    a = np.asarray(var.values if hasattr(var, "values") else var)

    # --- Case B: (N,) fixed-length strings ---
    if a.ndim == 1 and a.dtype.kind in ("S", "U"):
        out = []
        for x in a:
            if isinstance(x, (bytes, np.bytes_)):
                out.append(x.decode("utf-8", errors="ignore").strip())
            else:
                out.append(str(x).strip())
        return np.array(out, dtype=object)

    # --- Case C1: (N,) integers (rare) ---
    if a.ndim == 1 and a.dtype.kind in ("i", "u"):
        # interpret as bytes until 0-terminator if present
        out = []
        for val in a:
            out.append(chr(int(val)))
        return np.array(out, dtype=object)

    # From here: we expect 2D (N, STRINGLEN)
    if a.ndim != 2:
        # fallback
        return a.astype(str)

    # --- Case A: 2D unicode chars ---
    if a.dtype.kind == "U":
        return np.array(["".join(row).strip() for row in a], dtype=object)

    # --- Case A: 2D bytes chars (S1) ---
    if a.dtype.kind == "S":
        out = []
        for row in a:
            # row might already be a fixed-length bytes scalar in some weird cases
            if row.ndim == 0 and isinstance(row.item(), (bytes, np.bytes_)):
                out.append(row.item().decode("utf-8", errors="ignore").strip())
            else:
                out.append(b"".join(row).decode("utf-8", errors="ignore").strip())
        return np.array(out, dtype=object)

    # --- Case C2: 2D integers representing ASCII codes ---
    if a.dtype.kind in ("i", "u"):
        out = []
        for row in a:
            row = np.asarray(row)
            if (row == 0).any():
                row = row[:np.argmax(row == 0)]
            out.append(bytes(row.tolist()).decode("utf-8", errors="ignore").strip())
        return np.array(out, dtype=object)

    return a.astype(str)




def year_from_days_since_1908(days):
    """
    Convert 'days since 1908-01-01' to calendar year (UTC).
    """
    t0 = np.datetime64("1908-01-01T00:00:00")
    # days can be float with decimals
    t = t0 + (days * 86400.0).astype("timedelta64[s]")
    return pd.to_datetime(t).year


def basin_mask_station(lon, lat, basin):
    """
    Station-level basin mask using your basin tool.
    """
    xyz = np.zeros((len(lon), 3), dtype=float)
    xyz[:, 0] = lon
    xyz[:, 1] = lat
    good = np.isfinite(lon) & np.isfinite(lat)
    m = np.zeros(len(lon), dtype=bool)
    m[good] = basin.areinside(xyz[good])
    return m


def station_has_data(ds, varname, station_mask, use_qc=False):
    """
    Return boolean array (N_STATIONS,) indicating if a station has at least 1 valid sample for varname.
    Optionally require QC.
    """
    if varname not in ds.variables:
        return None

    data = ds[varname].values  # (N_STATIONS, N_SAMPLES)
    fill = ds[varname].attrs.get("_FillValue", None)
    valid = np.isfinite(data)
    if fill is not None:
        valid &= (data != fill)

    if use_qc:
        qc_name = f"{varname}_qc"
        if qc_name in ds.variables:
            qc = ds[qc_name].values  # same shape
            if GOOD_QC_ONLY:
                valid &= (qc == 48)  # good_quality only
            else:
                valid &= np.isin(qc, [48, 49, 52])  # good+unknown+questionable

    # station has any valid sample
    has_any = valid.any(axis=1)
    has_any &= station_mask
    return has_any


def summarize_file(platform, path):
    ds = xr.open_dataset(path, decode_times=False)

    # 1D station meta
    lon = ds["longitude"].values.astype(float)
    lat = ds["latitude"].values.astype(float)

    # station-level basin mask
    m_basin = basin_mask_station(lon, lat, basin)

    # decode cruise_id and station_id
    cruise = decode_odv_string(ds["cruise_id"])
    station = decode_odv_string(ds["station_id"]) if "station_id" in ds.variables else None

    days = ds["date_time"].values.astype(float)
    fill_time = ds["date_time"].attrs.get("_FillValue", None)

    bad = ~np.isfinite(days)

    # classic ODV fill
    if fill_time is not None:
        bad |= (days == fill_time)

    # IMPORTANT: treat near-zero as missing (common in some exports)
    # (0 days since 1908 => 1908-01-01, which creates fake early years)
    bad |= (days <= 1e-6)

    # sanity bounds (optional but recommended)
    bad |= (days < 0)
    bad |= (days > 365.25 * 200)   # ~ year 2108, prevents crazy outliers

    days = days.copy()
    days[bad] = np.nan

    years, days, t0 = years_from_date_time(ds)

    print(platform, "date_time stats:",
      "origin", t0,
      "min_days", np.nanmin(days), "max_days", np.nanmax(days),
      "n==0", int(np.sum(days == 0)),
      "n<1", int(np.sum((days >= 0) & (days < 1))),
      "n_nan", int(np.sum(~np.isfinite(days))))


    # Manca window (for Figure): 1908–1999
    valid_year = np.isfinite(years) & (years >= 1908) & (years <= 1999)
    cast_mask = m_basin & valid_year

    # Optionally require some variable exists (recommended if some stations are empty)
    # For MBT/XBT temperature is the key variable:
    if "Temperature" in ds.variables:
        hasT = station_has_data(ds, "Temperature", cast_mask, use_qc=USE_QC)
        cast_mask &= hasT


    n_casts = int(np.sum(cast_mask))
    n_datasets = int(pd.Series(cruise[cast_mask]).nunique(dropna=True))

    y_valid = years[cast_mask]
    y0 = int(np.nanmin(y_valid)) if np.isfinite(y_valid).any() else None
    y1 = int(np.nanmax(y_valid)) if np.isfinite(y_valid).any() else None
    period = f"{y0}–{y1}" if (y0 is not None and y1 is not None) else "NA"

    # yearly cast counts
    s = pd.Series(y_valid).dropna().astype(int)
    yearly = s.value_counts().sort_index()
    yearly.name = platform

    return {
        "platform": platform,
        "casts": n_casts,
        "datasets": n_datasets,
        "period": period,
        "yearly": yearly,
    }


# ---- RUN ALL PLATFORMS ----
summaries = []
for platform, path in files.items():
    print(f"Processing {platform}: {path}")
    summaries.append(summarize_file(platform, path))

# ---- TABLE 1 ----
table1 = pd.DataFrame([{
    "Data type": s["platform"],
    "Data sets no.": s["datasets"],
    "Profiles no.": s["casts"],
    "Period": s["period"],
} for s in summaries])

# total row
table1 = pd.concat([table1, pd.DataFrame([{
    "Data type": "Total",
    "Data sets no.": int(table1["Data sets no."].sum()),
    "Profiles no.": int(table1["Profiles no."].sum()),
    "Period": ""
}])], ignore_index=True)

print("\n=== Table 1 (Mediterranean mask applied) ===")
print(table1.to_string(index=False))

# ---- BUILD YEARLY DATAFRAME (from summaries) ----
# Each s["yearly"] is a pandas Series: index=Year, value=#casts
yearly_df = (
    pd.concat([s["yearly"] for s in summaries], axis=1)
      .fillna(0)
      .astype(int)
      .sort_index()
)

print("\nYearly cast counts (raw range):")
print(yearly_df.head())
print(yearly_df.tail())


for s in summaries:
    y = s["yearly"].index
    print(s["platform"], int(y.min()), int(y.max()), "nonzero years:", (s["yearly"]>0).sum())


# ---- YEARLY BARPLOT ----

# ---- LIMIT TO 20th CENTURY (Manca: 1900–2000) ----
yearly_df = yearly_df.loc[(yearly_df.index >= 1900) & (yearly_df.index <= 2000)]

# ---- Make continuous year axis ----
yearly_df = yearly_df.reindex(range(1900, 2001), fill_value=0)

# combine bathythermographs
yearly_plot = yearly_df.copy()
yearly_plot["MBT/XBT"] = yearly_plot.get("MBT", 0) + yearly_plot.get("XBT", 0)
yearly_plot = yearly_plot.drop(columns=[c for c in ["MBT", "XBT"] if c in yearly_plot.columns])

# ensure full year axis
yearly_plot = yearly_plot.reindex(range(1908, 2000), fill_value=0)

years = yearly_plot.index.values
bottle = yearly_plot.get("Bottle", pd.Series(0, index=years)).values
ctd    = yearly_plot.get("CTD",    pd.Series(0, index=years)).values
bathy  = yearly_plot.get("MBT/XBT",pd.Series(0, index=years)).values

fig, ax = plt.subplots(figsize=(16, 5))

# draw in the order that matches visibility
ax.bar(years, bathy,  width=0.8, facecolor="white", edgecolor="black", linewidth=0.8, label="MBT/XBT", zorder=1)
ax.bar(years, bottle, width=0.6, color="0.7",      edgecolor="black", linewidth=0.4, label="Bottle",  zorder=2)
ax.bar(years, ctd,    width=0.4, color="black",    edgecolor="black", linewidth=0.2, label="CTD",     zorder=3)

ax.set_xlim(1907.5, 1999.5)
ax.set_xlabel("Year")
ax.set_ylabel("Number of casts (stations)")
ax.legend(loc="upper right")

# sparse ticks like the paper
ticks = np.arange(1908, 2000, 3)   # try 5 if you want cleaner
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks], rotation=45, ha="right")

ax.grid(axis="y", linewidth=1.0)
ax.set_ylim(0, 9000) # consistent with Manca
fig.tight_layout()

out_png = "/Users/elena/Projects/IRB/Manca_Argo/Plots/Manca_like_Fig3_1908_1999.png"
fig.savefig(out_png, dpi=300)
plt.show()
print("Saved:", out_png)
