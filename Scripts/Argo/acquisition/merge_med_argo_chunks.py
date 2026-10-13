#!/usr/bin/env python3
# Reads : Data/Argo/Chunks/argo_med_phy*.nc
# Writes: Data/Argo/Merged/argo_med_merged_raw.nc, Data/Argo/Merged/merge_report.csv
"""
merge_med_chunks.py
--------------------

Merge all Mediterranean Argo chunk files into a single NetCDF and
produce a profile-count report.

- Reads all files matching argo_med_box*_Y*_Q*.nc
- Prints N_POINTS per file
- Sums total profiles
- Saves merged dataset
- Saves merge_report.csv for inspection

Usage:
    python merge_med_chunks.py
"""

import os
import glob
import pandas as pd
import xarray as xr
import numpy as np

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Scripts.paths import DATA

CHUNK_DIR    = str(DATA / "Argo" / "Chunks")
OUTDIR       = str(DATA / "Argo" / "Merged")
os.makedirs(OUTDIR, exist_ok=True)

OUT_NC   = os.path.join(OUTDIR, "argo_med_merged_raw.nc")
OUT_CSV  = os.path.join(OUTDIR, "merge_report.csv")


def main():

    print("== MERGING MEDITERRANEAN ARGO CHUNKS ==")
    print("Chunk directory:", CHUNK_DIR)

    #files = sorted(glob.glob(os.path.join(CHUNK_DIR, "argo_med_box*_Y*_Q*.nc")))
    files = sorted(glob.glob(os.path.join(CHUNK_DIR, "argo_med_phy*.nc")))
    if not files:
        print("No chunk files found. Check directory.")
        return

    print(f"Found {len(files)} chunk files.\n")

    report_rows = []
    datasets = []

    # -----------------------------------------
    # Process each chunk
    # -----------------------------------------
    for f in files:
        try:
            ds = xr.open_dataset(f)
        except Exception as e:
            print(f"[ERROR] Could not open file {f}: {e}")
            continue

        npts = ds.dims.get("N_POINTS", 0)

        # Store for report
        report_rows.append({
            "file": os.path.basename(f),
            "N_POINTS": npts
        })

        print(f"{os.path.basename(f):45s} -> {npts} points")

        if npts > 0:
            datasets.append(ds)

    # -----------------------------------------
    # Save report CSV
    # -----------------------------------------
    df_report = pd.DataFrame(report_rows)
    df_report.to_csv(OUT_CSV, index=False)

    print("\nSaved merge report to:", OUT_CSV)

    # Total profiles across all files
    total_points = df_report["N_POINTS"].sum()
    print(f"\nTOTAL POINTS DOWNLOADED (raw): {total_points}\n")

    # -----------------------------------------
    # Merge datasets
    # -----------------------------------------
    if not datasets:
        print("No datasets with N_POINTS > 0. Nothing to merge.")
        return

    print("Merging datasets...")

    # merge by concatenating along N_POINTS dimension
    ds_merged = xr.concat(datasets, dim="N_POINTS")

    # Sort by TIME to have chronological ordering
    time_var = "TIME" if "TIME" in ds_merged.variables else "time"
    order = np.argsort(ds_merged[time_var].values)
    ds_merged = ds_merged.isel(N_POINTS=order)

    ds_merged.attrs = {}  # avoid non-serializable global attrs

    print("Saving merged dataset to:", OUT_NC)
    ds_merged.to_netcdf(OUT_NC)
    print("Done.")

    # After ds_merged is defined and sorted
    plat_var = "PLATFORM_NUMBER" if "PLATFORM_NUMBER" in ds_merged.variables else "platform_number"
    cyc_var  = "CYCLE_NUMBER"    if "CYCLE_NUMBER"    in ds_merged.variables else "cycle_number"

    plat = ds_merged[plat_var].values
    cyc  = ds_merged[cyc_var].values

    profile_ids = pd.MultiIndex.from_arrays([plat, cyc])
    n_profiles = profile_ids.nunique()

    print(f"\nTOTAL DISTINCT PROFILES (casts) in merged Med dataset: {n_profiles}")



if __name__ == "__main__":
    main()
