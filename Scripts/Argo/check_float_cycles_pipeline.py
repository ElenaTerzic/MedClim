#!/usr/bin/env python3
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from argopy import DataFetcher
import argopy

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
WMO = 6903269

PROJECT = Path("/Users/elena/Projects/IRB/MedClim")

# raw / patched merged
MERGED_RAW = PROJECT / "Data/Argo/Merged/argo_med_merged_raw.nc"
# QC'ed merged
MERGED_QC  = PROJECT / "Data/Argo/Merged/argo_med_merged_qc.nc"
# flattened CSV
CSV_ALL    = PROJECT / "Data/Argo/Merged/argo_med_all_sorted.csv"
# with basins
CSV_BASINS = PROJECT / "Data/Argo/CSV/Argo_with_basins.csv"   # <- adjust if needed


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def cycles_from_nc(path, name):
    print(f"\n[NC] {name}: {path}")
    ds = xr.open_dataset(path)
    if "PLATFORM_NUMBER" not in ds:
        raise KeyError(f"{name} has no PLATFORM_NUMBER variable")

    mask = ds["PLATFORM_NUMBER"] == WMO
    npts = int(mask.sum().values)
    print(f"  Points for {WMO}: {npts}")

    if npts == 0:
        return np.array([], dtype=int)

    cyc = ds["CYCLE_NUMBER"].where(mask, drop=True).values.astype(int)
    cyc_unique = np.unique(cyc)
    print(f"  Cycles: {cyc_unique.min()} → {cyc_unique.max()}  (N={len(cyc_unique)})")
    return cyc_unique


def cycles_from_csv(path, name):
    print(f"\n[CSV] {name}: {path}")
    df = pd.read_csv(path, usecols=["PLATFORM_NUMBER", "CYCLE_NUMBER"])
    df_f = df[df["PLATFORM_NUMBER"] == WMO]
    print(f"  Rows for {WMO}: {len(df_f)}")
    if df_f.empty:
        return np.array([], dtype=int)

    cyc_unique = np.sort(df_f["CYCLE_NUMBER"].astype(int).unique())
    print(f"  Cycles: {cyc_unique.min()} → {cyc_unique.max()}  (N={len(cyc_unique)})")
    return cyc_unique


def report_diff(label, ref_cycles, here_cycles):
    missing = sorted(set(ref_cycles) - set(here_cycles))
    extra   = sorted(set(here_cycles) - set(ref_cycles))

    print(f"\n=== {label} vs ARGOPY reference ===")
    print(f"  Reference cycles (argopy): {ref_cycles.min()} → {ref_cycles.max()}  (N={len(ref_cycles)})")
    print(f"  Here cycles:               "
          f"{here_cycles.min() if len(here_cycles) else 'NONE'} → "
          f"{here_cycles.max() if len(here_cycles) else 'NONE'}  (N={len(here_cycles)})")
    print(f"  Missing here (in argopy, not here): {len(missing)}")
    if missing:
        print("    e.g.", missing[:20])
    print(f"  Extra here (not in argopy):        {len(extra)}")
    if extra:
        print("    e.g.", extra[:20])


def main():
    argopy.set_options(src="erddap", mode="standard")
    f = DataFetcher(ds="phy")

    print(f"[REF] Fetching float {WMO} from argopy ds='phy'...")
    ds_ref = f.float(str(WMO)).to_xarray()
    cyc_ref = np.unique(ds_ref["CYCLE_NUMBER"].values.astype(int))
    print(f"[REF] Cycles from argopy: {cyc_ref.min()} → {cyc_ref.max()}  (N={len(cyc_ref)})")

    # pipeline stages
    cyc_raw     = cycles_from_nc(MERGED_RAW, "MERGED_RAW")
    cyc_qc      = cycles_from_nc(MERGED_QC,  "MERGED_QC")
    cyc_csv_all = cycles_from_csv(CSV_ALL,    "CSV_ALL_SORTED")
    cyc_csv_bas = cycles_from_csv(CSV_BASINS, "CSV_ARGO_WITH_BASINS")

    # compare each stage to reference
    report_diff("MERGED_RAW",         cyc_ref, cyc_raw)
    report_diff("MERGED_QC",          cyc_ref, cyc_qc)
    report_diff("CSV_ALL_SORTED",     cyc_ref, cyc_csv_all)
    report_diff("CSV_ARGO_WITH_BASINS", cyc_ref, cyc_csv_bas)


if __name__ == "__main__":
    main()
