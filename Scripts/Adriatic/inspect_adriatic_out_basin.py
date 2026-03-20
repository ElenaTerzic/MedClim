#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect Adriatic processed long CSVs for basin labeling issues (e.g. 'out').

Goal:
- Identify which Adriatic source files contain basin values that do not map into DJ1/DJ2/DJ3.
- Dump the problematic rows to CSV for easy investigation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# -------------------------
# CONFIG
# -------------------------

ADR3_TO_MANCA_DJ: Dict[str, str] = {
    "nadr": "DJ1",
    "madr": "DJ2",
    "sadr": "DJ3",
    "north": "DJ1",
    "mid": "DJ2",
    "middle": "DJ2",
    "central": "DJ2",
    "south": "DJ3",
}

EXPECTED = {"DJ1", "DJ2", "DJ3"}

ADRIATIC_SOURCES = [
    ("CTD_HR_2019_2023_MFSD",
     "/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/CTD_HR_2019_2023_MFSD_long.csv"),
    ("RB_Northern_Adriatic_Transect_1979_2018",
     "/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/RB_Northern_Adriatic_Transect_1979_2018_long.csv"),
    ("Palagruza_Transect_1952_2018",
     "/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/Palagruza_Transect_1952_2018_long.csv"),
    ("IOR_Adriatic_database_1987_2019",
     "/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/IOR_Adriatic_database_1987_2019_long.csv"),
]

OUTDIR = Path("/Users/elena/Projects/IRB/MedClim/Diagnostics/Adriatic")
OUTDIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 400_000

# Candidate column names that might exist
BASIN_COL_CANDIDATES = ["BASIN_3ADR", "basin", "BASIN"]
TIME_COL_CANDIDATES = ["TIME", "time", "DATE_TIME", "datetime", "yyyy-mm-ddThh:mm:ss.sss"]
STATION_COL_CANDIDATES = ["STATION", "Station", "station", "STATION_NAME"]
DEPTH_COL_CANDIDATES = ["DEPTH_M", "Depth [m]", "DEPTH", "depth"]

LAT_CANDS = ["LAT", "lat", "Latitude", "Latitude [degrees_north]", "LATITUDE"]
LON_CANDS = ["LON", "lon", "Longitude", "Longitude [degrees_east]", "LONGITUDE"]


# -------------------------
# HELPERS
# -------------------------

def pick_first_existing(df_cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(df_cols)
    for c in candidates:
        if c in s:
            return c
    return None

def norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s

def remap_basin(raw: str) -> str:
    key = norm_str(raw).lower()
    if key == "":
        return ""
    return ADR3_TO_MANCA_DJ.get(key, norm_str(raw))  # if not mapped, keep original

def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

def describe_latlon(df: pd.DataFrame, latc: Optional[str], lonc: Optional[str]) -> str:
    if latc is None or lonc is None:
        return "lat/lon: (not found)"
    lat = pd.to_numeric(df[latc], errors="coerce")
    lon = pd.to_numeric(df[lonc], errors="coerce")
    if lat.notna().any() and lon.notna().any():
        return f"lat [{lat.min():.3f}, {lat.max():.3f}], lon [{lon.min():.3f}, {lon.max():.3f}]"
    return "lat/lon: present but non-numeric"

def summarize_bad_rows(df_bad: pd.DataFrame, basin_col: str, basin_mapped_col: str,
                       time_col: Optional[str], station_col: Optional[str]) -> None:
    print("  Bad rows:", len(df_bad))
    print("  Raw basin counts (top 15):")
    print(df_bad[basin_col].astype(str).value_counts().head(15).to_string())
    print("  Mapped basin counts (top 15):")
    print(df_bad[basin_mapped_col].astype(str).value_counts().head(15).to_string())

    if time_col and time_col in df_bad.columns:
        tt = safe_to_datetime(df_bad[time_col])
        if tt.notna().any():
            print(f"  Time span (bad rows): {tt.min()} ... {tt.max()}")

    if station_col and station_col in df_bad.columns:
        print("  Example stations (bad rows, top 15):")
        print(df_bad[station_col].astype(str).value_counts().head(15).to_string())

def iter_chunks(path: Path, usecols: List[str]):
    # dtype=str prevents nasty mixed-type surprises
    return pd.read_csv(path, chunksize=CHUNKSIZE, dtype=str, usecols=lambda c: c in set(usecols), low_memory=False)


# -------------------------
# MAIN
# -------------------------

def main():
    all_bad = []

    print("=== Adriatic basin inspector ===")
    print("Output dir:", OUTDIR)

    for src_name, src_path in ADRIATIC_SOURCES:
        p = Path(src_path)
        if not p.exists():
            print(f"\n[{src_name}] MISSING FILE:", p)
            continue

        # Peek header only (fast) to detect columns
        head = pd.read_csv(p, nrows=5, dtype=str, low_memory=False)
        cols = list(head.columns)

        basin_col = pick_first_existing(cols, BASIN_COL_CANDIDATES)
        time_col = pick_first_existing(cols, TIME_COL_CANDIDATES)
        station_col = pick_first_existing(cols, STATION_COL_CANDIDATES)
        depth_col = pick_first_existing(cols, DEPTH_COL_CANDIDATES)
        latc = pick_first_existing(cols, LAT_CANDS)
        lonc = pick_first_existing(cols, LON_CANDS)

        if basin_col is None:
            print(f"\n[{src_name}] No basin column found. Columns seen:", cols[:25], "...")
            continue

        # Choose minimal useful columns to read
        keep = [basin_col]
        for c in [time_col, station_col, depth_col, latc, lonc]:
            if c and c not in keep:
                keep.append(c)

        print(f"\n[{src_name}]")
        print("  file:", p)
        print("  basin col:", basin_col)
        print("  time col:", time_col)
        print("  station col:", station_col)
        print("  depth col:", depth_col)
        print("  lat/lon:", latc, lonc)

        # Accumulate value counts chunkwise to avoid full load
        raw_counts = {}
        mapped_counts = {}
        bad_rows_src = []

        for chunk in iter_chunks(p, usecols=keep):
            raw = chunk[basin_col].astype(str)

            mapped = raw.map(remap_basin)
            chunk["_basin_raw"] = raw.map(norm_str)
            chunk["_basin_mapped"] = mapped.map(norm_str)

            # update counts
            vc_raw = chunk["_basin_raw"].value_counts(dropna=False)
            vc_map = chunk["_basin_mapped"].value_counts(dropna=False)
            for k, v in vc_raw.items():
                raw_counts[k] = raw_counts.get(k, 0) + int(v)
            for k, v in vc_map.items():
                mapped_counts[k] = mapped_counts.get(k, 0) + int(v)

            # bad rows = mapped basin not in expected OR empty
            m_bad = (~chunk["_basin_mapped"].isin(list(EXPECTED))) | (chunk["_basin_mapped"] == "")
            if m_bad.any():
                bad_rows_src.append(chunk.loc[m_bad].copy())

        # Print summary counts
        raw_s = pd.Series(raw_counts).sort_values(ascending=False)
        map_s = pd.Series(mapped_counts).sort_values(ascending=False)

        print("  Raw BASIN_3ADR value_counts (top 15):")
        print(raw_s.head(15).to_string())
        print("  Mapped basin value_counts (top 15):")
        print(map_s.head(15).to_string())

        # Gather bad rows
        if bad_rows_src:
            bad = pd.concat(bad_rows_src, ignore_index=True)
            print("  >>> FOUND non-DJ basins after mapping!")
            print("  " + describe_latlon(bad, latc, lonc))
            summarize_bad_rows(bad, "_basin_raw", "_basin_mapped", time_col, station_col)

            bad["source_file"] = src_name
            all_bad.append(bad)

            out_bad = OUTDIR / f"adriatic_bad_basin_rows_{src_name}.csv"
            bad.to_csv(out_bad, index=False)
            print("  Wrote:", out_bad)
        else:
            print("  OK: all rows map to DJ1/DJ2/DJ3 (or file has no problematic basin labels).")

    # Merge all bad rows together
    if all_bad:
        bad_all = pd.concat(all_bad, ignore_index=True)
        out_all = OUTDIR / "adriatic_bad_basin_rows_ALL.csv"
        bad_all.to_csv(out_all, index=False)
        print("\n=== Combined bad rows ===")
        print("Total bad rows across Adriatic sources:", len(bad_all))
        print("Wrote:", out_all)
    else:
        print("\nNo bad rows found across Adriatic sources. (Then 'out' is likely created upstream in processing.)")

    print("\nDone.")


if __name__ == "__main__":
    main()
