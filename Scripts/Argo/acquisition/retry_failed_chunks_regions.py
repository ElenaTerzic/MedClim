#!/usr/bin/env python3
"""
retry_failed_chunks_regions.py
------------------------------

Read download_failures.csv produced by safe_download_med_argo_regions.py
and retry only those failed (region, month) tiles.

- Uses the SAME configuration (OUT_DIR, LOG_FILE, FAIL_FILE, Z_MIN, Z_MAX)
  as safe_download_med_argo_regions.py, by importing it.
- Reuses the logged lon/lat/time bounds exactly as originally requested.
- If a retry succeeds:
    * saves the chunk NetCDF in OUT_DIR
    * appends a success row to download_log.csv
    * does NOT keep it in download_failures.csv
- If it still fails:
    * keeps/updates the row in download_failures.csv

Usage examples:

    # retry all failed tiles
    python retry_failed_chunks_regions.py

    # retry only failures for one REGION_ID (e.g. 5)
    python retry_failed_chunks_regions.py --region-id 5
"""

from pathlib import Path
import argparse
import time
import traceback

import pandas as pd
import xarray as xr
from argopy import DataFetcher
import argopy


# --------------------------------------------------
# Import configuration from the main downloader
# --------------------------------------------------

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.Argo.acquisition import safe_download_med_argo_regions as cfg

OUT_DIR   = cfg.OUT_DIR
LOG_FILE  = cfg.LOG_FILE
FAIL_FILE = cfg.FAIL_FILE
Z_MIN     = cfg.Z_MIN
Z_MAX     = cfg.Z_MAX


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def append_log(row: dict):
    """Append a single success row to the main download_log.csv."""
    df = pd.DataFrame([row])
    if LOG_FILE.exists():
        df.to_csv(LOG_FILE, index=False, mode="a", header=False)
    else:
        df.to_csv(LOG_FILE, index=False)


def fetch_region_with_retry(fetcher, box, max_tries=4, sleep_sec=10):
    """
    Retry helper: try the same f.region(box) several times before giving up.
    """
    last_err = None
    for itry in range(1, max_tries + 1):
        try:
            ds = fetcher.region(box).to_xarray()
            return ds
        except Exception as e:
            last_err = e
            print(f"     [WARN] fetch attempt {itry}/{max_tries} failed: {e}")
            if itry < max_tries:
                time.sleep(sleep_sec)
    raise last_err


def main(region_id_filter: int | None):
    argopy.set_options(src="erddap", mode="standard")
    fetcher = DataFetcher(ds="phy")

    if not FAIL_FILE.exists():
        print(f"[INFO] No failure log found at {FAIL_FILE}. Nothing to retry.")
        return

    df_fail = pd.read_csv(FAIL_FILE)
    if df_fail.empty:
        print("[INFO] download_failures.csv is empty. Nothing to retry.")
        return

    # Optional: restrict to a single region_id
    if region_id_filter is not None:
        df_fail = df_fail[df_fail["region_id"] == region_id_filter]
        if df_fail.empty:
            print(f"[INFO] No failures for region_id={region_id_filter}")
            return

    print(f"[INFO] Retrying {len(df_fail)} failed tiles...")
    remaining_rows = []

    for idx, row in df_fail.iterrows():
        rid     = int(row["region_id"])
        yyyymm  = str(row["yyyymm"])
        lon_min = float(row["lon_min"])
        lon_max = float(row["lon_max"])
        lat_min = float(row["lat_min"])
        lat_max = float(row["lat_max"])
        t0      = str(row["t0"])
        t1      = str(row["t1"])

        # Reconstruct the output filename exactly like the main downloader
        outfile = OUT_DIR / (
            f"argo_med_phy_{yyyymm}_"
            f"R{rid}_"
            f"lon{lon_min:+05.1f}_{lon_max:+05.1f}_"
            f"lat{lat_min:+04.1f}_{lat_max:+04.1f}.nc"
        )

        print(
            f"\n[RETRY][R{rid}] {yyyymm}, "
            f"lon[{lon_min},{lon_max}], lat[{lat_min},{lat_max}], t={t0}–{t1}"
        )

        # Construct the region box
        box = [lon_min, lon_max, lat_min, lat_max, Z_MIN, Z_MAX, t0, t1]

        try:
            ds = fetch_region_with_retry(fetcher, box)
            npts = ds.dims.get("N_POINTS", 0)
            print(f"       -> N_POINTS = {npts}")

            # Strip attrs to avoid netCDF serialization issues
            ds = ds.copy()
            ds.attrs = {}
            for v in ds.variables:
                ds[v].attrs = {}

            # Save even if npts == 0 (same philosophy as main downloader)
            ds.to_netcdf(outfile)
            print(f"       [OK] saved {outfile.name}")

            # Log success to the main log
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
            print(f"       [FAIL] still failing: {e}")
            traceback.print_exc()
            # Keep this row with updated error text
            row_out = row.copy()
            row_out["error"] = repr(e)
            remaining_rows.append(row_out)

    # Rewrite the failure log with only the rows that still fail
    if remaining_rows:
        df_remain = pd.DataFrame(remaining_rows)
        df_remain.to_csv(FAIL_FILE, index=False)
        print(f"\n[INFO] {len(remaining_rows)} tiles still failing.")
        print(f"       Updated failure log: {FAIL_FILE}")
    else:
        # Clear failure log if all were fixed
        FAIL_FILE.unlink(missing_ok=True)
        print("\n[INFO] All previously failed tiles succeeded 🎉")
        print(f"       Removed {FAIL_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retry failed Med Argo region/month chunks."
    )
    parser.add_argument(
        "--region-id",
        type=int,
        default=None,
        help="Optional: retry only this REGION_ID (0..N-1).",
    )
    args = parser.parse_args()
    main(args.region_id)
