#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reads : Data/Seadatanet/*_mainland_special_excluded_no_argo_copies.csv, Data/Argo/CSV/Argo_with_basins_mainland_special_excluded.csv, Data/Adriatic/Processed/*_long_fixed_mainland_special_excluded.csv (SOURCES of multi_source_seasonal_climatology.py)
# Writes: Results/Trends/monthly_timeseries_all_bins.csv

"""
Build monthly time series per basin × depth_bin × var
directly from the multi-source streaming pipeline.

Uses:
  - SOURCES
  - iter_profiles_csv
  - scheme_for_basin
  - collect_cast_binned

from multi_source_seasonal_climatology.py

Output:
  monthly_timeseries_all_bins.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Scripts.paths import RESULTS

from Scripts.MultiPlatform.core import multi_source_seasonal_climatology as msc


# =========================
# SETTINGS
# =========================

OUT_CSV = str(RESULTS / "Trends" / "monthly_timeseries_all_bins.csv")

YEAR_MIN = 1950
YEAR_MAX = 2025

CHUNKSIZE = msc.CHUNKSIZE


# =========================
# MAIN
# =========================

def main():

    print("[INFO] Building monthly time series from raw sources")

    sumv = {}   # (basin, z, var, year, month) -> sum
    count = {}  # same key -> number of casts

    def accumulate(basin, z, var, year, month, value):
        key = (basin, float(z), var, int(year), int(month))
        sumv[key] = sumv.get(key, 0.0) + float(value)
        count[key] = count.get(key, 0) + 1

    for spec in msc.SOURCES:
        print(f"[INFO] Source: {spec.name}")

        for g in msc.iter_profiles_csv(spec, chunksize=CHUNKSIZE):

            basin = str(g[spec.basin_col].iloc[0]).strip().upper()
            if basin == "" or basin.lower() == "nan":
                continue

            # Parse time
            t = pd.to_datetime(g[spec.time_col].iloc[0], errors="coerce")
            if pd.isna(t):
                continue

            year = int(t.year)
            month = int(t.month)

            if year < YEAR_MIN or year > YEAR_MAX:
                continue

            scheme = msc.scheme_for_basin(basin)

            Tbin, Sbin, SIG0bin, ok, _, _ = msc.collect_cast_binned(g, spec, scheme)
            if not ok:
                continue

            zc = scheme.centers

            for i, z in enumerate(zc):

                vT = Tbin[i]
                if np.isfinite(vT):
                    accumulate(basin, z, "T", year, month, vT)

                vS = Sbin[i]
                if np.isfinite(vS):
                    accumulate(basin, z, "S", year, month, vS)

                vP = SIG0bin[i]
                if np.isfinite(vP):
                    accumulate(basin, z, "SIG0", year, month, vP)

    rows = []
    for (basin, z, var, year, month), sv in sumv.items():
        n = count[(basin, z, var, year, month)]
        val = sv / n if n > 0 else np.nan
        time = pd.Timestamp(year=year, month=month, day=1)
        rows.append((basin, z, var, time, year, month, val, n))

    df = pd.DataFrame(
        rows,
        columns=["basin", "z_m", "var", "time", "year", "month", "value", "n"]
    ).sort_values(["var", "basin", "z_m", "time"])

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"[OK] Wrote {OUT_CSV}")
    print(f"Rows: {len(df)} | Basins: {df['basin'].nunique()} | Bins: {df['z_m'].nunique()}")


if __name__ == "__main__":
    main()