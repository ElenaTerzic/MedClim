#!/usr/bin/env python3
"""
Convert merged Med Argo NetCDF to a profile-sorted CSV.

- Input:  merged + QC'ed Argo NetCDF (point dimension = N_POINTS)
- Output: CSV with one row per level, sorted by
          PLATFORM_NUMBER, CYCLE_NUMBER, PRES

This is designed to be compatible with your older Saline Lakes
workflow and with scripts like Argo_add_basins.py and compute_gradients.py.
"""

from pathlib import Path

import pandas as pd
import xarray as xr
import os


# ---------------------------------------------------------------------
# User config
# ---------------------------------------------------------------------

PROJECT_ROOT = Path("/Users/elena/Projects/IRB/MedClim")

IN_FILE = PROJECT_ROOT / "Data/Argo/Merged/argo_med_merged_qc.nc"
OUT_FILE = PROJECT_ROOT / "Data/Argo/Merged/argo_med_all_sorted.csv"


# Name of the point dimension in the NetCDF
POINT_DIM = "N_POINTS"

# Variables we really need for your saline-lake workflow
VARS_MANDATORY = [
    "PLATFORM_NUMBER",
    "CYCLE_NUMBER",
    "DIRECTION",
    "PRES",
    "TEMP",
    "PSAL",
    "LATITUDE",
    "LONGITUDE",
    "TIME",
    "N_POINTS",
    "SIG0",
    "PTEMP",
]

# QC / error variables that are nice-to-have but optional
VARS_OPTIONAL = [
    "DATA_MODE",
    "POSITION_QC",
    "PRES_ERROR", "PRES_QC",
    "PSAL_ERROR", "PSAL_QC",
    "TEMP_ERROR", "TEMP_QC",
    "TIME_QC",
]

# Final list to read from the netCDF: only include optional ones that exist
def vars_to_read(ds):
    missing_mand = [v for v in VARS_MANDATORY if v not in ds.variables]
    if missing_mand:
        raise KeyError(
            f"The following *mandatory* variables are missing in {IN_FILE}:\n{missing_mand}"
        )

    # Optional ones: take only those present
    opt_present = [v for v in VARS_OPTIONAL if v in ds.variables]
    return VARS_MANDATORY + opt_present


# ---------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------

def main():
    print("Opening NetCDF:")
    print(" ", IN_FILE)

    # decode_times=True is default, but explicit is nicer
    ds = xr.open_dataset(IN_FILE, decode_times=True)

    # Build the list of variables to read (mandatory + optional that exist)
    VARS = vars_to_read(ds)

    # Select only the columns we care about (smaller DataFrame, less RAM)
    ds_sel = ds[VARS]

    # Make sure POINT_DIM is a dimension
    if POINT_DIM not in ds_sel.dims:
        raise KeyError(
            f"Point dimension '{POINT_DIM}' not found in dataset dims: {list(ds_sel.dims)}"
        )

    # Put POINT_DIM first just for clarity (not strictly required)
    ds_sel = ds_sel.transpose(POINT_DIM, ...)

    print("\nFlattening to pandas DataFrame...")
    df = ds_sel.to_dataframe().reset_index()

    # -----------------------------------------------------------------
    # Ensure optional columns exist even if not present in the NetCDF
    # -----------------------------------------------------------------
    import numpy as np

    for v in VARS_OPTIONAL:
        if v not in df.columns:
            # QC/mode-like fields: fill with empty string
            if v.endswith("_QC") or v in ("DATA_MODE",):
                df[v] = ""
            else:
                # Error fields (e.g. PRES_ERROR): fill with NaN
                df[v] = np.nan

    # -----------------------------------------------------------------
    # 1) Normalise index / N_POINTS
    # -----------------------------------------------------------------
    if POINT_DIM in df.columns:
        df.rename(columns={POINT_DIM: "N_POINTS"}, inplace=True)
        # Make N_POINTS a plain integer index
        df["N_POINTS"] = df["N_POINTS"].astype("int64")
    else:
        # Fallback: create an explicit running index
        df.insert(0, "N_POINTS", range(len(df)))

    # -----------------------------------------------------------------
    # 2) Enforce types for key columns
    # -----------------------------------------------------------------
    # PLATFORM_NUMBER and CYCLE_NUMBER as integers
    df["PLATFORM_NUMBER"] = pd.to_numeric(df["PLATFORM_NUMBER"], errors="raise").astype(
        "int64"
    )
    df["CYCLE_NUMBER"] = pd.to_numeric(df["CYCLE_NUMBER"], errors="raise").astype(
        "int64"
    )

    # Pressure as float
    df["PRES"] = pd.to_numeric(df["PRES"], errors="raise")

    # TIME as datetime → string
    if pd.api.types.is_datetime64_any_dtype(df["TIME"]):
        t = df["TIME"]
    else:
        # In case TIME is numeric (e.g. days since ...)
        t = pd.to_datetime(df["TIME"], errors="coerce")

    # Format like your old CSV: "YYYY-MM-DD HH:MM:SS"
    df["TIME"] = t.dt.strftime("%Y-%m-%d %H:%M:%S")

    # Character-like QC / mode fields as stripped strings
    for col in ["DATA_MODE", "DIRECTION", "POSITION_QC", "PRES_QC", "PSAL_QC", "TEMP_QC", "TIME_QC"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # -----------------------------------------------------------------
    # 3) Drop obviously broken rows (very conservative)
    # -----------------------------------------------------------------
    before = len(df)
    df = df.dropna(subset=["PLATFORM_NUMBER", "CYCLE_NUMBER", "PRES", "LATITUDE", "LONGITUDE"])
    after = len(df)
    print(f"\nDropped {before - after} rows with missing core info "
          f"(kept {after} rows).")

    # -----------------------------------------------------------------
    # 4) Sort by (WMO, CYCLE, PRES) → contiguous profiles
    # -----------------------------------------------------------------
    print("\nSorting by PLATFORM_NUMBER, CYCLE_NUMBER, PRES...")
    df = df.sort_values(
        ["PLATFORM_NUMBER", "CYCLE_NUMBER", "PRES"]
    ).reset_index(drop=True)

    # -----------------------------------------------------------------
    # 5) Column order compatible with old workflow
    # -----------------------------------------------------------------
    cols_order = [
        "N_POINTS",
        "CYCLE_NUMBER",
        "DATA_MODE",
        "DIRECTION",
        "PLATFORM_NUMBER",
        "POSITION_QC",
        "PRES",
        "PRES_ERROR",
        "PRES_QC",
        "PSAL",
        "PSAL_ERROR",
        "PSAL_QC",
        "TEMP",
        "TEMP_ERROR",
        "TEMP_QC",
        "TIME_QC",
        "SIG0",
        "PTEMP",
        "LATITUDE",
        "LONGITUDE",
        "TIME",
    ]

    # Keep extra columns (if any) at the end
    cols = [c for c in cols_order if c in df.columns] + [
        c for c in df.columns if c not in cols_order
    ]
    df = df[cols]

    # -----------------------------------------------------------------
    # 6) Save to CSV
    # -----------------------------------------------------------------
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("\nWriting CSV to:")
    print(" ", OUT_FILE)
    df.to_csv(OUT_FILE, index=False)


    print("\nDone.")
    print(f"Rows written: {len(df)}")



# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
