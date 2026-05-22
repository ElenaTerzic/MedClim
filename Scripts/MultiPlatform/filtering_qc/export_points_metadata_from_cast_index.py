#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge cast_index_*.csv files into mapping-ready points tables.

Fixes vs v1:
- Skips empty/0-byte cast_index files (avoids pandas EmptyDataError)
- Preserves 'year' if present; only parses time if year missing
- Infers 'period' from parent folders and can optionally *validate* using year
- Writes TWO outputs by default: ..._20c.csv and ..._21c.csv (no mixing)
- Optional filters: basins, basin_modes (atomic/composed), include pooled/per_source

Edit USER SETTINGS below, then run:
    python export_points_metadata_from_cast_index_STANDALONE_v2.py
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================================
# USER SETTINGS
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT_STATS_DIR = str(PROJECT_ROOT / "Stats/MultiSource_Climatology_BINNED_ELIG")
OUT_PREFIX     = str(PROJECT_ROOT / "Stats/points_metadata_all_sources")  # will write _20c.csv and _21c.csv

# Keep only these basins (e.g. ["DJ1","DJ3"]) or None for all
BASINS_KEEP = None #["DJ1", "DJ3"]

# Keep only these basin modes (["atomic"], ["composed"], or ["atomic","composed"])
BASIN_MODES_KEEP = ["atomic", "composed"]   # recommend atomic for your plots; switch if needed

# Include which folders
INCLUDE_POOLED_RAW = True
INCLUDE_PER_SOURCE = True

# Optional: validate period using year and DROP mismatches (recommended True once you trust 'year')
DROP_PERIOD_YEAR_MISMATCH = False

# ==========================================================


def infer_period_from_path(path: Path) -> str | None:
    # Expect path parts include "20c_1950_1999" or "21c_2000_2025"
    for part in path.parts:
        if part.startswith("20c_"):
            return "20c_1950_1999"
        if part.startswith("21c_"):
            return "21c_2000_2025"
    return None


def infer_scope_from_path(path: Path) -> str:
    # pooled vs per_source
    parts = set(path.parts)
    if "POOLED_RAW" in parts:
        return "POOLED_RAW"
    if "per_source" in parts:
        return "per_source"
    return "unknown"


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        if path.stat().st_size == 0:
            return None
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            return None
        return df
    except pd.errors.EmptyDataError:
        return None


def standardize_cast_index(df: pd.DataFrame) -> pd.DataFrame:
    # We expect cast_index already has these fields from your climatology script:
    # source, basin, season, time, year, lon, lat, zmin_cast, zmax_cast, bin_scheme, basin_mode
    out = pd.DataFrame()

    # Basin & metadata
    for c in ["source", "basin", "season", "bin_scheme", "basin_mode"]:
        if c in df.columns:
            out[c] = df[c]

    # Lon/lat (keep as numeric)
    if "lon" in df.columns:
        out["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    if "lat" in df.columns:
        out["lat"] = pd.to_numeric(df["lat"], errors="coerce")

    # Time/year
    if "year" in df.columns:
        out["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "time" in df.columns:
        out["time"] = df["time"].astype(str)

    # If year missing but time exists, derive year
    if "year" not in out.columns or out["year"].isna().all():
        if "time" in out.columns:
            t = pd.to_datetime(out["time"], errors="coerce")
            out["year"] = t.dt.year.astype("Int64")

    # Depth range
    for c in ["zmin_cast", "zmax_cast"]:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    return out


def year_to_period(y: float | int | None) -> str | None:
    if y is None or (isinstance(y, float) and not np.isfinite(y)):
        return None
    try:
        y = int(y)
    except Exception:
        return None
    if 1950 <= y <= 1999:
        return "20c_1950_1999"
    if 2000 <= y <= 2025:
        return "21c_2000_2025"
    return None


def main():
    root = Path(ROOT_STATS_DIR)
    files = list(root.rglob("cast_index_*.csv"))
    if not files:
        print("No cast_index_*.csv files found under:", root)
        return

    print(f"Found {len(files)} cast index files.")
    frames = []

    for f in files:
        period = infer_period_from_path(f)
        if period is None:
            continue

        scope = infer_scope_from_path(f)
        if (scope == "POOLED_RAW" and not INCLUDE_POOLED_RAW) or (scope == "per_source" and not INCLUDE_PER_SOURCE):
            continue

        df = safe_read_csv(f)
        if df is None:
            # empty file -> skip silently
            continue

        df_std = standardize_cast_index(df)
        df_std["period"] = period
        df_std["scope"] = scope
        df_std["file"] = str(f)

        # Filters
        if BASIN_MODES_KEEP is not None and "basin_mode" in df_std.columns:
            df_std = df_std[df_std["basin_mode"].isin(BASIN_MODES_KEEP)]

        if BASINS_KEEP is not None and "basin" in df_std.columns:
            df_std = df_std[df_std["basin"].isin(BASINS_KEEP)]

        if DROP_PERIOD_YEAR_MISMATCH and "year" in df_std.columns:
            p_from_year = df_std["year"].apply(lambda x: year_to_period(x))
            ok = (p_from_year.isna()) | (p_from_year == df_std["period"])
            df_std = df_std[ok]

        if len(df_std):
            frames.append(df_std)

        print("Read:", f)

    if not frames:
        print("No data matched filters.")
        return

    merged = pd.concat(frames, ignore_index=True)

    # Write separate outputs by period (no mixing)
    for period in ["20c_1950_1999", "21c_2000_2025"]:
        sub = merged[merged["period"] == period].copy()
        out = Path(f"{OUT_PREFIX}_{period}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out, index=False)
        print("Wrote:", out, "| rows:", len(sub))

    # Also write combined (optional)
    out_all = Path(f"{OUT_PREFIX}_ALL.csv")
    merged.to_csv(out_all, index=False)
    print("Wrote:", out_all, "| rows:", len(merged))

    # Summary of lon/lat completeness per source
    if "lon" in merged.columns and "lat" in merged.columns:
        summ = (merged.groupby(["period","source"])
                      .agg(n=("source","size"),
                           frac_lon_na=("lon", lambda s: float(pd.isna(s).mean())),
                           frac_lat_na=("lat", lambda s: float(pd.isna(s).mean())))
                      .reset_index())
        print("\nLon/lat NA fractions:")
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
