#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebuild points metadata CSVs for map panels from the FILTERED source files
(*_mainland_special_excluded.csv).

Outputs
-------
For each basin mode:
    atomic
    composed

and for each period:
    ALL
    20c_1950_1999
    21c_2000_2025

write:

    points_metadata_all_sources_ALL_<mode>.csv
    points_metadata_all_sources_20c_1950_1999_<mode>.csv
    points_metadata_all_sources_21c_2000_2025_<mode>.csv

Columns
-------
source, basin, season, bin_scheme, basin_mode,
lon, lat, year, time, zmin_cast, zmax_cast,
period, scope, file

Notes
-----
- Designed to match the old points metadata structure used by plot_manca_all_basins.py
- Reads the already masked source CSVs, so excluded coastal/estuarine casts
  should no longer appear
- One row per cast/profile
- Chunked reading to keep memory manageable
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Iterable, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================

OUTDIR = Path("/Users/elena/Projects/IRB/MedClim/Stats")
CHUNKSIZE = 400_000

PERIODS = {
    "20c_1950_1999": (1950, 1999),
    "21c_2000_2025": (2000, 2025),
}

SEASONS = {
    "JFM": {1, 2, 3},
    "AMJ": {4, 5, 6},
    "JAS": {7, 8, 9},
    "OND": {10, 11, 12},
}

COMPOSED_BASINS: Dict[str, Dict[str, object]] = {
    "alb":  {"name": "Alboran",            "members": ["DS1"]},
    "west": {"name": "Western Med",        "members": ["DS2", "DS3", "DS4", "DI1"]},
    "nwm":  {"name": "North-Western Med",  "members": ["DF1", "DF2", "DF3", "DF4"]},
    "tyr":  {"name": "Tyrrhenian",         "members": ["DT1", "DT2", "DT3"]},
    "sic":  {"name": "Sicily Strait",      "members": ["DI3"]},
    "nadr": {"name": "Northern Adriatic",  "members": ["DJ1"]},
    "madr": {"name": "Mid Adriatic",       "members": ["DJ2"]},
    "sadr": {"name": "Southern Adriatic",  "members": ["DJ3"]},
    "ion":  {"name": "Ionian",             "members": ["DJ4", "DJ5", "DJ6", "DJ7", "DJ8"]},
    "lev":  {"name": "Levantine",          "members": ["DL1", "DL2", "DL3", "DL4", "DH3"]},
    "naeg": {"name": "Northern Aegean",    "members": ["DH1"]},
    "saeg": {"name": "Southern Aegean",    "members": ["DH2"]},
}

INCLUDE_UNMAPPED_IN_COMPOSED = False
ADRIATIC_BASINS = {"DJ1", "DJ2", "DJ3", "NADR", "MADR", "SADR"}

SCOPE = "POOLED_RAW"


# =============================================================================
# SOURCE DEFINITIONS
# =============================================================================

@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    time_col: str
    basin_col: str
    depth_col: str
    lon_col: Optional[str]
    lat_col: Optional[str]
    cast_id_cols: List[str]
    add_date_to_cast_id: bool = True
    time_format: Optional[str] = None
    basin_remap: Optional[Dict[str, str]] = None


SOURCES: List[SourceSpec] = [
    SourceSpec(
        name="SDN",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins_mainland_special_excluded_no_argo_copies.csv"),
        time_col="yyyy-mm-ddThh:mm:ss.sss",
        basin_col="basin",
        depth_col="Depth [m]",
        lon_col="Longitude [degrees_east]",
        lat_col="Latitude [degrees_north]",
        cast_id_cols=[
            "LOCAL_CDI_ID||LOCAL_CDI_ID:METAVAR:INDEXED_TEXT",
            "Cruise",
            "Station:METAVAR:INDEXED_TEXT",
        ],
        add_date_to_cast_id=False,
    ),
    SourceSpec(
        name="ARGO",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Argo/CSV/Argo_with_basins_mainland_special_excluded.csv"),
        time_col="TIME",
        time_format="%Y-%m-%dT%H:%M:%S",
        basin_col="basin",
        depth_col="PRES",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        cast_id_cols=["PLATFORM_NUMBER", "CYCLE_NUMBER"],
        add_date_to_cast_id=False,
    ),
    SourceSpec(
        name="CTD_HR_2019_2023_MFSD",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/CTD_HR_2019_2023_MFSD_long_fixed_mainland_special_excluded.csv"),
        time_col="TIME",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        cast_id_cols=["STATION"],
        add_date_to_cast_id=True,
    ),
    SourceSpec(
        name="RB_Northern_Adriatic_Transect_1979_2018",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/RB_Northern_Adriatic_Transect_1979_2018_long_fixed_mainland_special_excluded.csv"),
        time_col="TIME",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        cast_id_cols=["STATION"],
        add_date_to_cast_id=True,
    ),
    SourceSpec(
        name="Palagruza_Transect_1952_2018",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/Palagruza_Transect_1952_2018_long_fixed_mainland_special_excluded.csv"),
        time_col="TIME",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        cast_id_cols=["STATION"],
        add_date_to_cast_id=True,
    ),
    SourceSpec(
        name="IOR_Adriatic_database_1987_2019",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/IOR_Adriatic_database_1987_2019_long_fixed_mainland_special_excluded.csv"),
        time_col="TIME",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        cast_id_cols=["STATION", "CAST_IDX"],
        add_date_to_cast_id=True,
    ),
]


# =============================================================================
# HELPERS
# =============================================================================

def month_to_season(m: int) -> Optional[str]:
    for s, months in SEASONS.items():
        if m in months:
            return s
    return None


def normalize_basin_value(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "nan"):
        return ""
    return s.upper()


def apply_basin_remap(basin: str, remap: Optional[Dict[str, str]]) -> str:
    if remap is None:
        return basin
    return remap.get(basin, basin)


def map_basin_id(basin: str, basin_mode: str) -> str:
    b = str(basin).strip().upper()

    if basin_mode == "atomic":
        return b

    if basin_mode != "composed":
        raise ValueError(f"Unknown basin_mode: {basin_mode}")

    for comp_id, meta in COMPOSED_BASINS.items():
        members = [str(x).upper() for x in meta["members"]]
        if b in members:
            return comp_id.upper()

    return b if INCLUDE_UNMAPPED_IN_COMPOSED else ""


def scheme_for_basin(basin: str) -> str:
    return "ADRIATIC" if str(basin).strip().upper() in ADRIATIC_BASINS else "DEFAULT"


def first_existing_column(df: pd.DataFrame, spec: str) -> Optional[str]:
    """
    spec can be:
      "A" or "A||B||C"
    """
    alts = [x.strip() for x in spec.split("||")]
    for col in alts:
        if col in df.columns:
            return col
    return None


def safe_to_datetime(s: pd.Series, fmt: Optional[str] = None) -> pd.Series:
    if fmt:
        return pd.to_datetime(s, format=fmt, errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def build_cast_uid_series(df: pd.DataFrame, spec: SourceSpec) -> pd.Series:
    parts = []

    for colspec in spec.cast_id_cols:
        col = first_existing_column(df, colspec)
        if col is None:
            parts.append(pd.Series([""] * len(df), index=df.index))
        else:
            parts.append(df[col].astype(str).fillna("").str.strip())

    uid = parts[0].copy()
    for p in parts[1:]:
        uid = uid + "||" + p

    if spec.add_date_to_cast_id and spec.time_col in df.columns:
        t = safe_to_datetime(df[spec.time_col], spec.time_format)
        uid = uid + "||" + t.dt.strftime("%Y-%m-%d").fillna("")

    return uid


def required_columns_for_source(spec: SourceSpec) -> List[str]:
    cols = [spec.time_col, spec.basin_col, spec.depth_col]
    if spec.lon_col:
        cols.append(spec.lon_col)
    if spec.lat_col:
        cols.append(spec.lat_col)

    for c in spec.cast_id_cols:
        alts = [x.strip() for x in c.split("||")]
        cols.extend(alts)

    out = []
    seen = set()
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def finalize_cast(g: pd.DataFrame, spec: SourceSpec, basin_mode: str) -> Optional[dict]:
    if g.empty:
        return None

    basin_raw = g[spec.basin_col].iloc[0] if spec.basin_col in g.columns else ""
    basin = normalize_basin_value(basin_raw)
    basin = apply_basin_remap(basin, spec.basin_remap)
    if basin == "":
        return None

    basin_mapped = map_basin_id(basin, basin_mode)
    if basin_mapped == "":
        return None

    t0 = pd.to_datetime(g[spec.time_col].iloc[0], errors="coerce")
    if pd.isna(t0):
        return None

    year = int(t0.year)
    season = month_to_season(int(t0.month))
    if season is None:
        return None

    z = pd.to_numeric(g[spec.depth_col], errors="coerce").to_numpy(float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None

    lon = np.nan
    lat = np.nan
    if spec.lon_col and spec.lon_col in g.columns:
        lonv = pd.to_numeric(g[spec.lon_col], errors="coerce")
        if lonv.notna().any():
            lon = float(lonv.dropna().iloc[0])

    if spec.lat_col and spec.lat_col in g.columns:
        latv = pd.to_numeric(g[spec.lat_col], errors="coerce")
        if latv.notna().any():
            lat = float(latv.dropna().iloc[0])

    return {
        "source": spec.name,
        "basin": basin_mapped,
        "season": season,
        "bin_scheme": scheme_for_basin(basin_mapped),
        "basin_mode": basin_mode,
        "lon": lon,
        "lat": lat,
        "year": year,
        "time": str(t0),
        "zmin_cast": float(np.nanmin(z)),
        "zmax_cast": float(np.nanmax(z)),
    }


def iter_cast_rows(spec: SourceSpec, basin_mode: str, chunksize: int = CHUNKSIZE) -> Iterable[dict]:
    usecols = [c for c in required_columns_for_source(spec) if c is not None]
    carry = None

    for chunk in pd.read_csv(spec.path, usecols=lambda c: c in usecols, chunksize=chunksize):
        chunk = chunk.copy()
        chunk["_cast_uid"] = build_cast_uid_series(chunk, spec)

        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = None

        if chunk.empty:
            continue

        # hold back last cast in case it continues into next chunk
        last_uid = chunk["_cast_uid"].iloc[-1]
        mask_last = chunk["_cast_uid"] == last_uid

        carry = chunk.loc[mask_last].copy()
        ready = chunk.loc[~mask_last].copy()

        if not ready.empty:
            for _, g in ready.groupby("_cast_uid", sort=False):
                row = finalize_cast(g, spec, basin_mode)
                if row is not None:
                    yield row

    if carry is not None and not carry.empty:
        for _, g in carry.groupby("_cast_uid", sort=False):
            row = finalize_cast(g, spec, basin_mode)
            if row is not None:
                yield row


def build_points_df(basin_mode: str) -> pd.DataFrame:
    rows = []
    for spec in SOURCES:
        if not spec.path.exists():
            print(f"[WARN] Missing file, skipping: {spec.path}")
            continue

        print(f"[INFO] Reading {spec.name}: {spec.path}")
        n0 = len(rows)

        for row in iter_cast_rows(spec, basin_mode=basin_mode):
            row["scope"] = SCOPE
            row["file"] = str(spec.path)
            rows.append(row)

        print(f"[INFO]   added {len(rows) - n0} casts")

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Keep only finite coords for plotting metadata
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df = df[np.isfinite(df["lon"]) & np.isfinite(df["lat"]) & np.isfinite(df["year"])].copy()

    # Deduplicate cautiously
    dedup_cols = ["source", "basin", "time", "lon", "lat", "zmin_cast", "zmax_cast", "basin_mode"]
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    # Sort
    df = df.sort_values(["source", "basin", "year", "time", "lon", "lat"]).reset_index(drop=True)

    # ALL period
    df["period"] = "ALL"

    return df


def write_period_files(df_all: pd.DataFrame, basin_mode: str) -> None:
    if df_all.empty:
        print(f"[WARN] No rows for basin_mode={basin_mode}")
        return

    out_all = OUTDIR / f"points_metadata_all_sources_ALL_{basin_mode}.csv"
    df_all.to_csv(out_all, index=False)
    print(f"[OUTPUT] {out_all}")

    for period_name, (ymin, ymax) in PERIODS.items():
        d = df_all[(df_all["year"] >= ymin) & (df_all["year"] <= ymax)].copy()
        d["period"] = period_name

        out = OUTDIR / f"points_metadata_all_sources_{period_name}_{basin_mode}.csv"
        d.to_csv(out, index=False)
        print(f"[OUTPUT] {out}")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    for basin_mode in ["atomic", "composed"]:
        print("=" * 90)
        print(f"[INFO] Building points metadata for basin_mode={basin_mode}")
        df_all = build_points_df(basin_mode=basin_mode)
        write_period_files(df_all, basin_mode=basin_mode)

    print("=" * 90)
    print("[DONE]")


if __name__ == "__main__":
    main()