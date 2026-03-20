import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import Basins

# -------- SETTINGS --------
nc_path = "/Users/elena/Projects/IRB/Manca_Argo/Data/Medatlas/NetCDF/allTS_data.nc"
basin = Basins.worldseas.med

YEAR_MIN = 1908
YEAR_MAX = 1999

SAVE_PLOT = True
out_png = "/Users/elena/Projects/IRB/Manca_Argo/Plots/allTS_Med_yearly_casts_1908_1999.png"
# -------------------------


def decode_odv_string(var):
    a = np.asarray(var.values if hasattr(var, "values") else var)

    if a.ndim == 1 and a.dtype.kind in ("S", "U"):
        out = []
        for x in a:
            if isinstance(x, (bytes, np.bytes_)):
                out.append(x.decode("utf-8", errors="ignore").strip())
            else:
                out.append(str(x).strip())
        return np.array(out, dtype=object)

    if a.ndim == 2 and a.dtype.kind == "U":
        return np.array(["".join(row).strip() for row in a], dtype=object)

    if a.ndim == 2 and a.dtype.kind == "S":
        out = []
        for row in a:
            out.append(b"".join(row).decode("utf-8", errors="ignore").strip())
        return np.array(out, dtype=object)

    if a.ndim == 2 and a.dtype.kind in ("i", "u"):
        out = []
        for row in a:
            row = np.asarray(row)
            if (row == 0).any():
                row = row[:np.argmax(row == 0)]
            out.append(bytes(row.tolist()).decode("utf-8", errors="ignore").strip())
        return np.array(out, dtype=object)

    return a.astype(str)


def years_from_date_time(ds):
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
        raise ValueError(f"Cannot parse date_time units: {units}")

    date0, time0 = m.group(1), m.group(2)
    t0 = pd.Timestamp(f"{date0}T{time0}Z")

    dt = t0 + pd.to_timedelta(days, unit="D")
    years = pd.DatetimeIndex(dt).year.to_numpy(dtype=float)
    years = np.where(np.isfinite(days), years, np.nan)

    return years, days, t0


def basin_mask_station(lon, lat, basin):
    xyz = np.zeros((len(lon), 3), dtype=float)
    xyz[:, 0] = lon
    xyz[:, 1] = lat
    good = np.isfinite(lon) & np.isfinite(lat)
    m = np.zeros(len(lon), dtype=bool)
    m[good] = basin.areinside(xyz[good])
    return m


def station_has_any_valid(ds, varname, station_mask):
    if varname not in ds.variables:
        return station_mask.copy()

    data = ds[varname].values  # (N_STATIONS, N_SAMPLES)
    fill = ds[varname].attrs.get("_FillValue", None)

    valid = np.isfinite(data)
    if fill is not None:
        valid &= (data != fill)

    has_any = valid.any(axis=1)
    return station_mask & has_any


def station_has_n_valid(ds, varname, station_mask, nmin=2):
    if varname not in ds.variables:
        return np.zeros(len(station_mask), dtype=bool)

    data = ds[varname].values  # (N_STATIONS, N_SAMPLES)
    fill = ds[varname].attrs.get("_FillValue", None)

    valid = np.isfinite(data)
    if fill is not None:
        valid &= (data != fill)

    n_valid = valid.sum(axis=1)
    return station_mask & (n_valid >= nmin)


def main():
    ds = xr.open_dataset(nc_path, decode_times=False)

    # station meta
    lon = ds["longitude"].values.astype(float)
    lat = ds["latitude"].values.astype(float)

    m_basin = basin_mask_station(lon, lat, basin)

    cruise = decode_odv_string(ds["cruise_id"]) if "cruise_id" in ds.variables else None
    station = decode_odv_string(ds["station_id"]) if "station_id" in ds.variables else None

    years, days, t0 = years_from_date_time(ds)
    print("allTS time origin:", t0)
    print("days==0:", int(np.sum(days == 0)), "days NaN:", int(np.sum(~np.isfinite(days))))

    valid_year = np.isfinite(years) & (years >= YEAR_MIN) & (years <= YEAR_MAX)

    # If you want to avoid “origin spikes” from days==0 used as missing:
    valid_year &= (days != 0)

    # ============================================================
    # CAST VALIDITY FOR allTS (stations with Temperature OR Salinity)
    # ============================================================

    base_mask = m_basin & valid_year

    # Manca Fig.3-like: count stations that have Temperature (includes MBT/XBT + CTD + Bottle)
    cast_mask = station_has_any_valid(ds, "Temperature", base_mask)

    # Optional (light) robustness: uncomment if you want to exclude 1-point casts
    # cast_mask = station_has_n_valid(ds, "Temperature", cast_mask, nmin=2)

    print("After Temperature-only filter:", int(np.sum(cast_mask)))


    # ============================================================
    # RAW COUNTS (no dedup)
    # ============================================================

    n_casts_raw = int(np.sum(cast_mask))
    n_datasets_raw = int(pd.Series(cruise[cast_mask]).nunique()) if cruise is not None else np.nan

    y_raw = pd.Series(years[cast_mask]).dropna().astype(int)
    yearly_raw = y_raw.value_counts().sort_index()

    print("\n=== allTS RAW ===")
    print("Casts (stations):", f"{n_casts_raw:,}")
    print("Datasets (unique cruise_id):", f"{n_datasets_raw:,}" if np.isfinite(n_datasets_raw) else "NA")

    # ============================================================
    # DEDUPLICATION (check for duplicate stations inside allTS)
    # ============================================================

    lon_r  = np.round(lon, 3)
    lat_r  = np.round(lat, 3)
    days_r = np.round(days, 3)

    dfk = pd.DataFrame({
        "cruise":  cruise[cast_mask] if cruise is not None else "",
        "station": station[cast_mask] if station is not None else "",
        "lon":     lon_r[cast_mask],
        "lat":     lat_r[cast_mask],
        "days":    days_r[cast_mask],
        "year":    years[cast_mask].astype(int),
    })

    dfk_dedup = dfk.drop_duplicates(subset=["cruise", "station", "lon", "lat", "days"])

    n_casts = len(dfk_dedup)
    n_datasets = int(dfk_dedup["cruise"].nunique()) if cruise is not None else np.nan

    yearly = dfk_dedup["year"].value_counts().sort_index()

    print("\n=== allTS DEDUP ===")
    print("Casts (stations):", f"{n_casts:,}")
    print("Datasets (unique cruise_id):", f"{n_datasets:,}" if np.isfinite(n_datasets) else "NA")
    print("Duplicates removed:", f"{n_casts_raw - n_casts:,}")

    print("\n=== allTS (Mediterranean mask + year window) ===")
    print("Casts (stations):", f"{n_casts:,}")
    print("Datasets (unique cruise_id):", f"{n_datasets:,}" if np.isfinite(n_datasets) else "NA")
    print("Period:", f"{int(yearly.index.min())}–{int(yearly.index.max())}" if len(yearly) else "NA")

    # plot
    if SAVE_PLOT and len(yearly):
        yr = yearly.reindex(range(YEAR_MIN, YEAR_MAX + 1), fill_value=0)

        fig, ax = plt.subplots(figsize=(16, 5))
        ax.bar(yr.index.values, yr.values, width=0.8, edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of casts (stations)")
        ax.set_title("Medatlas II allTS — Mediterranean basin — yearly casts")

        ticks = np.arange(YEAR_MIN, YEAR_MAX + 1, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], rotation=45, ha="right")

        ax.grid(axis="y", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.show()
        print("Saved:", out_png)


if __name__ == "__main__":
    main()
