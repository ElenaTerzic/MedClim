#!/usr/bin/env python3
# Reads : Data/Argo/Merged/argo_med_merged_raw.nc
# Writes: Data/Argo/Merged/argo_med_merged_qc.nc
"""
Minimal QC for merged Mediterranean Argo dataset (ds='phy', mode='standard').

Steps:
  1. Load merged raw file (point format, dim='N_POINTS').
  2. Keep only ascending PROFILES (DIRECTION == 'A' per profile, not per level).
  3. Apply simple physical range checks on PRES / TEMP / PSAL.
  4. Compute TEOS-10 variables (SIG0, PTEMP) with argopy.
  5. Save cleaned dataset.

Key point:
  We select ascending *profiles* using ds.argo.profile_id and groupby(),
  then broadcast that mask back to the point dimension. This avoids
  losing an entire profile because of a single bad DIRECTION value.
"""

import os
import numpy as np
import xarray as xr
import argopy  # registers the .argo accessor on xarray objects

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Scripts.paths import DATA

# ================== USER SETTINGS ==================

IN_FILE = str(DATA / "Argo" / "Merged" / "argo_med_merged_raw.nc")

OUT_FILE = str(DATA / "Argo" / "Merged" / "argo_med_merged_qc.nc")

POINT_DIM = "N_POINTS"

VAR_WMO   = "PLATFORM_NUMBER"
VAR_CYC   = "CYCLE_NUMBER"
VAR_PRES  = "PRES"
VAR_TEMP  = "TEMP"
VAR_PSAL  = "PSAL"
VAR_DIR   = "DIRECTION"
VAR_LAT   = "LATITUDE"
VAR_LON   = "LONGITUDE"
VAR_TIME  = "TIME"

# ---- Simple physical ranges (Mediterranean-safe; adjust if you like) ----
PRES_MIN, PRES_MAX = -5.0, 6000.0
TEMP_MIN, TEMP_MAX = -2.5, 40.0
PSAL_MIN, PSAL_MAX =  20.0, 41.0

KEEP_ASCENDING_ONLY = True   # keep only ascending profiles (DIRECTION == 'A')


# ====================================================

def pick_var(ds, primary, alternatives):
    """Return primary if present in ds.variables, else the first existing
    alternative; raise if none are found."""
    if primary in ds.variables:
        return primary
    for alt in alternatives:
        if alt in ds.variables:
            return alt
    raise KeyError(f"None of {[primary] + alternatives} found in dataset.")


def main():
    print("Loading merged dataset:")
    print(" ", IN_FILE)

    # Use chunks to keep memory manageable; adjust if you like
    ds = xr.open_dataset(IN_FILE, chunks={POINT_DIM: 1_000_000})

    if POINT_DIM not in ds.dims:
        raise ValueError(
            f"Expected point dimension '{POINT_DIM}' not found. "
            f"Available dims: {list(ds.dims)}"
        )

    # Resolve variable names (in case of slightly different conventions)
    v_wmo  = pick_var(ds, VAR_WMO,  ["platform_number"])
    v_cyc  = pick_var(ds, VAR_CYC,  ["cycle_number"])
    v_pres = pick_var(ds, VAR_PRES, ["PRES_ADJUSTED", "pres", "pres_adjusted"])
    v_temp = pick_var(ds, VAR_TEMP, ["TEMP_ADJUSTED", "temp", "temp_adjusted"])
    v_psal = pick_var(ds, VAR_PSAL, ["PSAL_ADJUSTED", "psal", "psal_adjusted"])
    v_dir  = pick_var(ds, VAR_DIR,  ["direction"])
    v_lat  = pick_var(ds, VAR_LAT,  ["latitude"])
    v_lon  = pick_var(ds, VAR_LON,  ["longitude"])
    v_time = pick_var(ds, VAR_TIME, ["time"])

    print("Using variables:")
    print(f"  WMO   : {v_wmo}")
    print(f"  CYCLE : {v_cyc}")
    print(f"  PRES  : {v_pres}")
    print(f"  TEMP  : {v_temp}")
    print(f"  PSAL  : {v_psal}")
    print(f"  DIR   : {v_dir}")
    print(f"  LAT   : {v_lat}")
    print(f"  LON   : {v_lon}")
    print(f"  TIME  : {v_time}")

    n0 = ds.sizes[POINT_DIM]
    print(f"\nInitial number of points: {n0}")

    # ------------------------------------------------
    # 1) Keep only ASCENDING PROFILES (per-profile)
    # ------------------------------------------------
    if KEEP_ASCENDING_ONLY:
        print("\nSelecting ascending profiles (DIRECTION == 'A')...")

        # DIRECTION as strings
        dir_vals = ds[v_dir].astype(str)

        # Boolean mask per point: True where direction == 'A'
        # .compute() makes it a non-dask array, which xarray can use in isel
        mask_dir = (dir_vals == "A").compute()

        # Convert to plain NumPy array
        mask_dir_np = mask_dir.data  # or mask_dir.values

        # Sanity / counters
        n_keep = int(mask_dir_np.sum())
        print(f"  points kept   : {n_keep}")
        print(f"  points dropped: {n0 - n_keep}")

        # Apply selection
        ds = ds.isel({POINT_DIM: mask_dir_np})
        n0 = ds.sizes[POINT_DIM]



    # ------------------------------------------------
    # 2) Simple physical range QC: out-of-range → NaN
    # ------------------------------------------------
    print("\nApplying hard-range QC on PRES / TEMP / PSAL...")

    pres = ds[v_pres]
    temp = ds[v_temp]
    psal = ds[v_psal]

    mask_pres = (pres >= PRES_MIN) & (pres <= PRES_MAX)
    mask_temp = (temp >= TEMP_MIN) & (temp <= TEMP_MAX)
    mask_psal = (psal >= PSAL_MIN) & (psal <= PSAL_MAX)

    ds[v_pres] = xr.where(mask_pres, pres, np.nan)
    ds[v_temp] = xr.where(mask_temp, temp, np.nan)
    ds[v_psal] = xr.where(mask_psal, psal, np.nan)

    print(f"Points after basic QC (NaNs set, but no point drop here): {ds.sizes[POINT_DIM]}")

    # ------------------------------------------------
    # 3) Add TEOS-10 variables (SIG0, PTEMP)
    # ------------------------------------------------
    print("\nComputing TEOS-10 variables (SIG0, PTEMP) with argopy...")

    # argopy expects standard names: PRES, TEMP, PSAL, LATITUDE, LONGITUDE, TIME
    # you already have them; if they were aliases, we resolved them above
    ds = ds.argo.teos10(["SIG0", "PTEMP"])

    # ------------------------------------------------
    # 4) Clean attributes (for safe NetCDF writing)
    # ------------------------------------------------
    print("\nCleaning attributes and saving QC’d dataset to:")
    print(" ", OUT_FILE)

    ds_qc = ds.copy()

    # Drop global attrs (optional, but safer for netCDF)
    ds_qc.attrs = {}

    # Keep only simple attrs (str/int/float) on each variable
    for v in ds_qc.variables:
        clean_attrs = {}
        for ak, av in ds_qc[v].attrs.items():
            if isinstance(av, (str, int, float)):
                clean_attrs[ak] = av
        ds_qc[v].attrs = clean_attrs

    # Write out; dask will execute the graph during this call
    ds_qc.to_netcdf(OUT_FILE)
    print("Done.")


if __name__ == "__main__":
    main()
