#!/usr/bin/env python3
# Reads : Argo GDAC via argopy (network)
# Writes: Data/Argo/Chunks/argo_med_phy_<yyyymm>_R<region>_lon*_lat*.nc, download_log.csv, download_failures.csv
"""
Safer Med Argo downloader using argopy ds="phy".

- Time: monthly chunks
- Space: multiple lon/lat tiles (regions)
- Each region has a REGION_ID so you can run tiles in parallel:
    python safe_download_med_argo_regions.py --region-id 0
    python safe_download_med_argo_regions.py --region-id 1
    ...
- Now we have 24 regions, 4 lat bands x 6 lon bands
- Logs successes and failures to CSV
- Saves one NetCDF per tile (year, month, lon, lat)
"""

from pathlib import Path
import argparse
import pandas as pd
import xarray as xr
from argopy import DataFetcher
import argopy
import traceback

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Scripts.paths import DATA

# --------------------- CONFIG ------------------------

OUT_DIR = DATA / "Argo" / "Chunks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE  = OUT_DIR / "download_log.csv"
FAIL_FILE = OUT_DIR / "download_failures.csv"

YEAR_START = 2025
YEAR_END   = 2025   # inclusive

TIME_PAD_DAYS = 1  # overlap a day before and after each month

LAT_PAD = 0.1   # degrees of overlap in latitude
LON_PAD = 0.1   # degrees of overlap in longitude


# Lat / lon bands for tiling the Med
# You can tweak these if you like (e.g. more lat bands).
LAT_BANDS = [
    (30.0, 34.0),
    (34.0, 38.0),
    (38.0, 42.0),
    (42.0, 46.0),
]

LON_BANDS = [
    (-6.0,  1.0),
    (1.0 ,  8.0),
    (8.0 , 15.0),
    (15.0, 22.0),
    (22.0, 29.0),
    (29.0, 36.0),
]

Z_MIN, Z_MAX = 0.0, 5000.0

# ------------------- REGION TILES --------------------

def build_region_tiles():
    """
    Build a list of region tiles with a bit of spatial overlap.
    Each entry: (region_id, lon_min, lon_max, lat_min, lat_max)
    """
    tiles = []
    rid = 0

    # overall domain limits so we don't expand beyond the Med box
    LON_MIN_GLOBAL = min(l[0] for l in LON_BANDS)
    LON_MAX_GLOBAL = max(l[1] for l in LON_BANDS)
    LAT_MIN_GLOBAL = min(l[0] for l in LAT_BANDS)
    LAT_MAX_GLOBAL = max(l[1] for l in LAT_BANDS)

    for lon_min, lon_max in LON_BANDS:
        # overlapped lon:
        lon_min_pad = max(LON_MIN_GLOBAL, lon_min - LON_PAD)
        lon_max_pad = min(LON_MAX_GLOBAL, lon_max + LON_PAD)
        for lat_min, lat_max in LAT_BANDS:
            # overlapped lat:
            lat_min_pad = max(LAT_MIN_GLOBAL, lat_min - LAT_PAD)
            lat_max_pad = min(LAT_MAX_GLOBAL, lat_max + LAT_PAD)

            tiles.append((rid, lon_min_pad, lon_max_pad, lat_min_pad, lat_max_pad))
            rid += 1
    return tiles


REGION_TILES = build_region_tiles()

# ------------------- HELPERS -------------------------

def year_months(year: int):
    """Return list of (YYYYMM, start_date, end_date) for each month.

    We use:
      start = first day of month minus TIME_PAD_DAYS
      end   = first day of next month plus TIME_PAD_DAYS

    The idea: over-cover in time and deduplicate later by (platform, cycle).
    """
    dates = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="MS")
    out = []
    for t0, t1 in zip(dates[:-1], dates[1:]):
        yyyymm = t0.strftime("%Y%m")

        start = (t0 - pd.Timedelta(days=TIME_PAD_DAYS)).strftime("%Y-%m-%d")
        end   = (t1 + pd.Timedelta(days=TIME_PAD_DAYS)).strftime("%Y-%m-%d")

        out.append((yyyymm, start, end))
    return out



def init_logs():
    """Create empty log files if they don't exist."""
    if not LOG_FILE.exists():
        df = pd.DataFrame(columns=[
            "region_id", "yyyymm",
            "lon_min", "lon_max", "lat_min", "lat_max",
            "t0", "t1", "n_points", "outfile"
        ])
        df.to_csv(LOG_FILE, index=False)

    if not FAIL_FILE.exists():
        df = pd.DataFrame(columns=[
            "region_id", "yyyymm",
            "lon_min", "lon_max", "lat_min", "lat_max",
            "t0", "t1", "error"
        ])
        df.to_csv(FAIL_FILE, index=False)

def append_log(row: dict):
    df = pd.DataFrame([row])
    df.to_csv(LOG_FILE, index=False, mode="a", header=False)

def append_fail(row: dict):
    df = pd.DataFrame([row])
    df.to_csv(FAIL_FILE, index=False, mode="a", header=False)

def print_region_map():
    print("Available REGION_ID tiles:")
    for rid, lon_min, lon_max, lat_min, lat_max in REGION_TILES:
        print(f"  REGION_ID {rid}: "
              f"lon[{lon_min:5.1f}, {lon_max:5.1f}], "
              f"lat[{lat_min:4.1f}, {lat_max:4.1f}]")

# ------------------- MAIN DOWNLOAD -------------------

def main(region_id: int | None):
    argopy.set_options(src="erddap", mode="standard")
    f = DataFetcher(ds="phy")

    init_logs()

    # Select which tiles to process
    if region_id is None:
        tiles = REGION_TILES
        print("[INFO] No REGION_ID specified: downloading ALL regions.")
    else:
        tiles = [t for t in REGION_TILES if t[0] == region_id]
        if not tiles:
            raise SystemExit(f"Invalid REGION_ID {region_id}.")
        print(f"[INFO] Restricting to REGION_ID {region_id}")

    print_region_map()
    print()

    for year in range(YEAR_START, YEAR_END + 1):
        for yyyymm, t0, t1 in year_months(year):
            for rid, lon_min, lon_max, lat_min, lat_max in tiles:

                box = [lon_min, lon_max,
                       lat_min, lat_max,
                       Z_MIN, Z_MAX,
                       t0, t1]

                outfile = OUT_DIR / (
                    f"argo_med_phy_{yyyymm}_"
                    f"R{rid}_"
                    f"lon{lon_min:+05.1f}_{lon_max:+05.1f}_"
                    f"lat{lat_min:+04.1f}_{lat_max:+04.1f}.nc"
                )

                # Skip if already downloaded
                if outfile.exists():
                    print(f"[SKIP][R{rid}] {yyyymm} -> {outfile.name} exists")
                    continue

                print(f"[DL][R{rid}] {yyyymm}, "
                      f"lon[{lon_min},{lon_max}], "
                      f"lat[{lat_min},{lat_max}], "
                      f"t={t0}–{t1}")
                try:
                    ds = f.region(box).to_xarray()
                    npts = ds.dims.get("N_POINTS", 0)
                    print(f"       -> N_POINTS = {npts}")

                    if npts == 0:
                        print(f"       !! WARNING: zero points in this tile. "
                                 "If this is unexpected, check ERDDAP or try again later.")

                    # Save even if npts==0 (so the tile is marked as done)
                    ds.attrs = {}
                    ds.to_netcdf(outfile)

                    append_log({
                        "region_id": rid,
                        "yyyymm": yyyymm,
                        "lon_min": lon_min,
                        "lon_max": lon_max,
                        "lat_min": lat_min,
                        "lat_max": lat_max,
                        "t0": t0,
                        "t1": t1,
                        "n_points": npts,
                        "outfile": str(outfile),
                    })

                except Exception as e:
                    print(f"       !! FAILED: {e}")
                    append_fail({
                        "region_id": rid,
                        "yyyymm": yyyymm,
                        "lon_min": lon_min,
                        "lon_max": lon_max,
                        "lat_min": lat_min,
                        "lat_max": lat_max,
                        "t0": t0,
                        "t1": t1,
                        "error": repr(e),
                    })
                    traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Safe Med Argo downloader with REGION_ID tiles."
    )
    parser.add_argument(
        "--region-id", type=int, default=None,
        help="Which REGION_ID tile to download (0..N-1). "
             "If omitted, all regions are processed."
    )
    args = parser.parse_args()
    main(args.region_id)
