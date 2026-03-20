#!/usr/bin/env python3
"""
SeaDataNet temporal maps per Manca basin (2x3 per basin):
- Rows: 20c (1950-1999), 21c (2000-2019)
- Cols: CTD, Bottle, XBT (MBT merged into XBT)
Points colored by date.

Reads the BIG "Seadatanet.profiles_only_with_basins.csv" which already has a 'basin' column.
Robustly auto-detects SeaDataNet-style column names like:
- 'Longitude [degrees_east]'
- 'Latitude [degrees_north]'
- 'Instrument / gear type:METAVAR:INDEXED_TEXT'
- 'yyyy-mm-ddThh:mm:ss.sss'

Output: one PNG per basin, 2x3 panel layout.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import Basins

# -----------------------------
# USER SETTINGS
# -----------------------------
INPUT_CSV = "/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins.csv"
OUTDIR    = "/Users/elena/Projects/IRB/MedClim/Plots/SeaDataNet/Basins/Temporal_2x3/"

BASIN_LIST = [
    Basins.manca.df2, Basins.manca.df3, Basins.manca.df4, Basins.manca.ds2, Basins.manca.df1,
    Basins.manca.ds1, Basins.manca.ds3, Basins.manca.ds4, Basins.manca.dt1, Basins.manca.dt2,
    Basins.manca.dt3, Basins.manca.di1, Basins.manca.di3, Basins.manca.dj1, Basins.manca.dj2,
    Basins.manca.dj3, Basins.manca.dj6, Basins.manca.dj4, Basins.manca.dj7, Basins.manca.dj8,
    Basins.manca.dj5, Basins.manca.dh1, Basins.manca.dh2, Basins.manca.dh3, Basins.manca.dl1,
    Basins.manca.dl2, Basins.manca.dl3, Basins.manca.dl4
]

PERIODS = {
    "20c_1950_1999": (datetime(1950, 1, 1), datetime(1999, 12, 31, 23, 59, 59)),
    "21c_2000_2019": (datetime(2000, 1, 1), datetime(2019, 12, 31, 23, 59, 59)),
}

PLATFORMS = ["CTD", "Bottle", "XBT"]  # 3 columns in the 2x3 plot
CHUNKSIZE = 500_000                   # tune if needed


# -----------------------------
# HELPERS
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def find_col(cols, candidates=None, contains_all=None):
    """
    Find a column in `cols` by:
      1) exact match in `candidates`
      2) substring match: all strings in `contains_all` must be present (case-insensitive)
    """
    cols_list = list(cols)
    cols_set = set(cols_list)

    if candidates:
        for c in candidates:
            if c in cols_set:
                return c

    if contains_all:
        contains_all = [s.lower() for s in contains_all]
        for c in cols_list:
            cl = c.lower()
            if all(s in cl for s in contains_all):
                return c

    return None


def find_time_col(cols):
    """Pick a matching time column used in SeaDataNet exports."""
    candidates = [
        "TIME", "time", "date_time", "datetime", "DATE",
        "yyyy-mm-ddThh:mm:ss.sss",
        "yyyy-mm-ddThh:mm:ss",
        "yyyy-mm-ddThh:mm:ss.s",
    ]
    for c in candidates:
        if c in cols:
            return c

    # fallback: anything that resembles the SeaDataNet pattern
    for c in cols:
        cl = c.lower()
        if ("yyyy" in cl) and ("mm" in cl) and ("dd" in cl) and ("hh" in cl):
            return c

    raise ValueError(f"Could not find a time column. Available columns: {list(cols)}")


def platform_from_instrument(instr):
    """Map SeaDataNet instrument strings into CTD/Bottle/XBT."""
    if instr is None or (isinstance(instr, float) and np.isnan(instr)):
        return "Other"
    s = str(instr).upper()

    # CTD family
    if "CTD" in s or "ROSETTE CTD" in s:
        return "CTD"

    # XBT / MBT family
    if "XBT" in s or "MBT" in s or "BATHYTHERM" in s:
        return "XBT"

    # Bottle / sampling / rosette / niskin / hydrocast keywords
    if ("BOTTLE" in s or "NISKIN" in s or "ROSETTE" in s or "WATER SAMPL" in s
        or "HYDROCAST" in s or "SAMPLER" in s):
        return "Bottle"

    return "Other"


def _date2num_from_datetime64(arr_dt64):
    """
    Convert numpy datetime64[ns] (or pandas datetime-like) to matplotlib date numbers robustly.
    Works for:
      - numpy datetime64 arrays
      - pandas Series[datetime64]
      - pandas DatetimeIndex
    """
    if arr_dt64 is None:
        return np.array([], dtype=float)

    # Normalize to pandas datetime-like (can be DatetimeIndex)
    dt = pd.to_datetime(arr_dt64, errors="coerce")

    # Drop NaT in a way that works for both Series and DatetimeIndex
    if isinstance(dt, pd.Series):
        dt = dt.dropna()
        if dt.empty:
            return np.array([], dtype=float)
        py = dt.dt.to_pydatetime()
    else:
        # DatetimeIndex
        dt = dt[~dt.isna()]
        if len(dt) == 0:
            return np.array([], dtype=float)
        py = dt.to_pydatetime()

    return mdates.date2num(py)



# -----------------------------
# MAIN READ (single pass, chunked)
# -----------------------------
def resolve_columns():
    """
    Read only header to resolve actual column names in the BIG CSV.
    Returns dict with keys: basin, lon, lat, instr, time
    """
    header = pd.read_csv(INPUT_CSV, nrows=0)
    cols = header.columns

    time_col = find_time_col(cols)

    lon_col = find_col(
        cols,
        candidates=["longitude", "Longitude", "LONGITUDE", "Longitude [degrees_east]"],
        contains_all=["longitude"]
    )
    lat_col = find_col(
        cols,
        candidates=["latitude", "Latitude", "LATITUDE", "Latitude [degrees_north]"],
        contains_all=["latitude"]
    )
    instr_col = find_col(
        cols,
        candidates=[
            "instrument", "Instrument", "INSTRUMENT",
            "Instrument / gear type:METAVAR:INDEXED_TEXT",
        ],
        contains_all=["instrument", "gear"]
    )
    basin_col = find_col(
        cols,
        candidates=["basin", "BASIN"],
        contains_all=["basin"]
    )

    resolved = {"basin": basin_col, "lon": lon_col, "lat": lat_col, "instr": instr_col, "time": time_col}

    missing = [k for k, v in resolved.items() if (v is None or v not in cols)]
    if missing:
        raise ValueError(
            f"Could not resolve required columns: {missing}\n"
            f"Resolved mapping: {resolved}\n"
            f"Available columns: {list(cols)}"
        )

    print("[INFO] Column mapping resolved:")
    for k, v in resolved.items():
        print(f"  {k:>5s} -> {v}")

    return resolved


def load_points_by_basin_period_platform():
    """
    Stream the big CSV in chunks, keep only lon/lat/time for basin+period+platform.
    Store only unique casts (assuming input is already profiles-only; if not, we still plot points).
    """
    ensure_dir(OUTDIR)

    col = resolve_columns()

    # Storage: basin -> period -> platform -> arrays
    store = {
        basin.abbrev: {
            per: {p: {"lon": [], "lat": [], "t": []} for p in PLATFORMS}
            for per in PERIODS
        }
        for basin in BASIN_LIST
    }
    basin_set = set(store.keys())

    usecols = [col["basin"], col["lon"], col["lat"], col["instr"], col["time"]]

    # Iterate chunks
    for chunk in pd.read_csv(
        INPUT_CSV,
        usecols=usecols,
        chunksize=CHUNKSIZE,
        low_memory=False
    ):
        # Basin filter early
        chunk = chunk[chunk[col["basin"]].isin(basin_set)]
        if chunk.empty:
            continue

        # Parse time + numeric lon/lat
        chunk[col["time"]] = pd.to_datetime(chunk[col["time"]], errors="coerce")
        chunk[col["lon"]]  = pd.to_numeric(chunk[col["lon"]], errors="coerce")
        chunk[col["lat"]]  = pd.to_numeric(chunk[col["lat"]], errors="coerce")

        chunk = chunk.dropna(subset=[col["time"], col["lon"], col["lat"], col["basin"]])
        if chunk.empty:
            continue

        # Platform class
        chunk["platform3"] = chunk[col["instr"]].map(platform_from_instrument)
        chunk = chunk[chunk["platform3"].isin(PLATFORMS)]
        if chunk.empty:
            continue

        # Split by period
        for per, (t0, t1) in PERIODS.items():
            m = chunk[col["time"]].between(t0, t1)
            cc = chunk.loc[m, :]
            if cc.empty:
                continue

            # Append to store
            for bname, sub_b in cc.groupby(col["basin"], sort=False):
                for plat, sub_p in sub_b.groupby("platform3", sort=False):
                    d = store[bname][per][plat]
                    d["lon"].append(sub_p[col["lon"]].to_numpy(dtype=float))
                    d["lat"].append(sub_p[col["lat"]].to_numpy(dtype=float))
                    d["t"].append(sub_p[col["time"]].to_numpy(dtype="datetime64[ns]"))

    # Concatenate lists
    for b in store:
        for per in PERIODS:
            for plat in PLATFORMS:
                d = store[b][per][plat]
                if d["lon"]:
                    d["lon"] = np.concatenate(d["lon"])
                    d["lat"] = np.concatenate(d["lat"])
                    d["t"]   = np.concatenate(d["t"])
                else:
                    d["lon"] = np.array([], dtype=float)
                    d["lat"] = np.array([], dtype=float)
                    d["t"]   = np.array([], dtype="datetime64[ns]")

    return store


# -----------------------------
# PLOTTING
# -----------------------------
def plot_basin_2x3(basin, data):
    fig = plt.figure(figsize=(16, 9))
    proj = ccrs.PlateCarree()

    # Determine extent from all panels
    lons_all, lats_all = [], []
    for per in PERIODS:
        for plat in PLATFORMS:
            lons_all.append(data[per][plat]["lon"])
            lats_all.append(data[per][plat]["lat"])

    has_any = any(x.size for x in lons_all)
    if has_any:
        lons_all = np.concatenate([x for x in lons_all if x.size])
        lats_all = np.concatenate([x for x in lats_all if x.size])

        xmin, xmax = np.nanmin(lons_all), np.nanmax(lons_all)
        ymin, ymax = np.nanmin(lats_all), np.nanmax(lats_all)

        dx = (xmax - xmin) * 0.15 if xmax > xmin else 0.5
        dy = (ymax - ymin) * 0.15 if ymax > ymin else 0.5
        extent = [xmin - dx, xmax + dx, ymin - dy, ymax + dy]
    else:
        extent = [-6, 37, 30, 46]  # fallback

    # Keep track of row axes for row-wise colorbars
    row_axes = {0: [], 1: []}

    # Fixed color range per row = the period bounds (good for comparability)
    period_items = list(PERIODS.items())

    for irow, (per_name, (t0, t1)) in enumerate(period_items):
        vmin = mdates.date2num(t0)
        vmax = mdates.date2num(t1)

        for icol, plat in enumerate(PLATFORMS):
            ax = fig.add_subplot(2, 3, irow * 3 + icol + 1, projection=proj)
            row_axes[irow].append(ax)

            ax.set_extent(extent, crs=proj)
            ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)

            lon = data[per_name][plat]["lon"]
            lat = data[per_name][plat]["lat"]
            tt  = data[per_name][plat]["t"]

            n = int(lon.size)

            if n > 0:
                tnum = _date2num_from_datetime64(tt)
                # If some dates failed parsing, mask lon/lat accordingly
                if tnum.size != n:
                    # rebuild using pandas Series mask
                    dt = pd.to_datetime(tt, errors="coerce")
                    good = dt.notna().to_numpy()
                    lon2 = lon[good]
                    lat2 = lat[good]
                    tnum2 = mdates.date2num(dt[good].dt.to_pydatetime())
                    ax.scatter(lon2, lat2, c=tnum2, s=18, transform=proj, vmin=vmin, vmax=vmax, cmap="jet")
                    n = int(lon2.size)
                else:
                    ax.scatter(lon, lat, c=tnum, s=18, transform=proj, vmin=vmin, vmax=vmax, cmap="jet")
            ax.set_title(f"{per_name} — {plat} (n={n})", fontsize=11)

        # Row-wise colorbar (even if no data, still shows the period scale)
        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="jet")
        sm.set_array([])

        cb = fig.colorbar(sm, ax=row_axes[irow], fraction=0.025, pad=0.02)
        cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"SeaDataNet — {basin.name} ({basin.abbrev}) — temporal coverage", fontsize=14)

    out = os.path.join(OUTDIR, f"SDN_{basin.abbrev}_{basin.name.replace(' ', '_')}_2x3_temporal.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {out}")


def main():
    ensure_dir(OUTDIR)
    store = load_points_by_basin_period_platform()

    for basin in BASIN_LIST:
        plot_basin_2x3(basin, store[basin.abbrev])


if __name__ == "__main__":
    main()
