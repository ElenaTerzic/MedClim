#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_standard_depths_DJ3.py

Check "standard depth grid" alignment for DJ3 basin only, per file/folder, optionally per platform.

SeaDataNet-specific handling:
- depth:  "Depth [m]"
- profile id: "LOCAL_CDI_ID"
- platform text: "Instrument / gear type:METAVAR:INDEXED_TEXT"
- maps to 4 platform classes only: TSG, MBT/XBT, Bottle, CTD (everything else excluded)

Outputs:
- standard_depths_summary_DJ3.csv
- standard_depths_profile_match_DJ3.csv  (optional; if a profile id column exists)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# EDIT HERE
# =============================================================================

TARGET_BASIN = "DJ3"

STD_CENTERS = np.array([
    5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200,
    250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,
    1200, 1300, 1400, 1500, 1750, 2000, 2500, 3000,
    3500, 4000, 4500, 5000
], dtype=float)

STD_TOL = 0.5

# SDN: keep only these (matches your plot classes)
VALID_PLATFORMS = ["TSG", "MBT/XBT", "Bottle", "CTD"]

# Inputs (files or directories). Directories are scanned recursively.
INPUTS: List[Path] = [
    Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins.csv"),
    Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/CTD_HR_2019_2023_MFSD_long_fixed.csv"),
    Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/IOR_Adriatic_database_1987_2019_long_fixed.csv"),
    Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/RB_Northern_Adriatic_Transect_1979_2018_long_fixed.csv"),
    Path("/Users/elena/Projects/IRB/MedClim/Data/Argo/CSV/Argo_with_basins.csv"),
]

OUTDIR = Path("/Users/elena/Projects/IRB/MedClim/Plots/Adriatic/DEPTH_STATS")  # set to your output folder if desired

COMPUTE_PROFILE_MATCH = True
SPLIT_BY_PLATFORM_IF_AVAILABLE = True

MAX_UNIQUE_DEPTHS_TO_STORE = 80


# =============================================================================
# Column heuristics
# =============================================================================

# SeaDataNet exact column names (from your head.csv)
SDN_DEPTH_COL = "Depth [m]"
SDN_PROFILE_ID_COL = "LOCAL_CDI_ID"
SDN_PLATFORM_TEXT_COL = "Instrument / gear type:METAVAR:INDEXED_TEXT"

DEPTH_CANDIDATES = [
    SDN_DEPTH_COL,
    "depth", "DEPTH", "Depth", "z", "Z", "z_m", "DEPTH_M", "depth_m", "DEPH",
    "pres", "PRES", "pressure", "PRESSURE", "PRES_REL", "PRES_ADJ"
]

PROFILE_ID_CANDIDATES = [
    SDN_PROFILE_ID_COL,
    "profile_id", "PROFILE_ID", "pid", "PID", "cast_id", "CAST_ID", "cast",
    "station", "STATION", "station_id", "STATION_ID",
    "wmo", "WMO", "wmo_id", "CYCLE_NUMBER", "cycle_number", "cycle", "CYCLE"
]

PLATFORM_CANDIDATES = [
    SDN_PLATFORM_TEXT_COL,
    "platform_class", "PLATFORM_CLASS", "platform_type", "PLATFORM_TYPE",
    "instrument", "INSTRUMENT", "gear", "GEAR",
    "instrument_type", "INSTRUMENT_TYPE", "platform", "PLATFORM",
    "ptype", "PTYPE", "edmo_platform", "EDMO_PLATFORM", "nstr", "NSTR"
]

# basin column candidates (we still auto-detect by content, but this helps)
BASIN_CANDIDATES = [
    "basin", "BASIN", "basin_abbrev", "BASIN_ABBREV", "basin_id", "BASIN_ID",
    "Basin", "BASIN_NAME", "basin_name"
]


# =============================================================================
# Helpers
# =============================================================================

@dataclass
class DetectedCols:
    depth: str
    profile_id: Optional[str]
    platform: Optional[str]
    depth_kind: str  # "depth" or "pressure"


def _pick_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # case-insensitive fallback
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def detect_columns(df: pd.DataFrame) -> DetectedCols:
    depth_col = _pick_first_existing(df, DEPTH_CANDIDATES)
    if depth_col is None:
        raise ValueError("Could not detect a depth/pressure column. Add it to DEPTH_CANDIDATES.")

    profile_col = _pick_first_existing(df, PROFILE_ID_CANDIDATES)
    platform_col = _pick_first_existing(df, PLATFORM_CANDIDATES)

    dname = depth_col.lower()
    if "pres" in dname or "pressure" in dname:
        kind = "pressure"
    else:
        kind = "depth"

    return DetectedCols(depth=depth_col, profile_id=profile_col, platform=platform_col, depth_kind=kind)


def list_supported_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        exts = {".csv", ".tsv", ".txt", ".parquet", ".pq", ".nc", ".nc4", ".cdf"}
        return sorted([p for p in path.rglob("*") if p.suffix.lower() in exts and p.is_file()])
    return []


def read_any_table(fp: Path) -> pd.DataFrame:
    suf = fp.suffix.lower()

    if suf in (".csv", ".tsv", ".txt"):
        # try common delimiters (SDN sometimes uses ; elsewhere)
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(fp, sep=sep, low_memory=False)
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
        return pd.read_csv(fp, sep=None, engine="python", low_memory=False)

    if suf in (".parquet", ".pq"):
        return pd.read_parquet(fp)

    if suf in (".nc", ".nc4", ".cdf"):
        import xarray as xr
        ds = xr.open_dataset(fp)
        return ds.to_dataframe().reset_index()

    raise ValueError(f"Unsupported file type: {fp}")


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def nearest_distance_to_std(vals: np.ndarray, std: np.ndarray) -> np.ndarray:
    if vals.size == 0:
        return vals
    d = np.abs(vals.reshape(-1, 1) - std.reshape(1, -1))
    return d.min(axis=1)


def detect_basin_column_by_value(df: pd.DataFrame, target: str) -> Optional[str]:
    """
    Find a column containing basin codes by:
      - checking BASIN_CANDIDATES first
      - else scanning object columns for presence of `target` (exact match after strip)
    """
    # 1) prefer known names
    cand = _pick_first_existing(df, BASIN_CANDIDATES)
    if cand is not None:
        s = df[cand].astype(str).str.strip()
        if (s == target).any():
            return cand

    # 2) scan small-cardinality object columns
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in obj_cols:
        s = df[c].astype(str).str.strip()
        # quick presence check
        if (s == target).any():
            return c

    return None


def filter_to_basin(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, Optional[str], int]:
    basin_col = detect_basin_column_by_value(df, target)
    if basin_col is None:
        return df.iloc[0:0].copy(), None, 0
    sub = df[df[basin_col].astype(str).str.strip() == target].copy()
    return sub, basin_col, int(len(sub))


def sdn_platform_class_from_instrument(df: pd.DataFrame,
                                       instrument_col: str,
                                       cruise_col: Optional[str] = "Cruise") -> pd.Series:
    """
    Map SDN instrument/gear strings to your 4 classes:
    TSG, MBT/XBT, Bottle, CTD. Everything else -> NA.
    """
    inst = df[instrument_col].astype(str).str.upper().fillna("")
    out = pd.Series(["NA"] * len(df), index=df.index, dtype="object")

    # CTD
    out[inst.str.contains(r"\bCTD\b", regex=True)] = "CTD"

    # Bottle (common SDN variants)
    out[inst.str.contains(r"\bBOT\b|\bBOTTLE\b|ROSETTE|NISKIN", regex=True)] = "Bottle"

    # MBT/XBT
    out[inst.str.contains(r"\bXBT\b|\bMBT\b", regex=True)] = "MBT/XBT"

    # TSG
    out[inst.str.contains(r"\bTSG\b|THERMOSALINOGRAPH", regex=True)] = "TSG"

    # Optional fallback: Cruise tags like *_XBT, *_BOT, etc.
    if cruise_col is not None and cruise_col in df.columns:
        cru = df[cruise_col].astype(str).str.upper().fillna("")
        na = out.eq("NA")
        out.loc[na & cru.str.contains(r"_CTD\b")] = "CTD"
        out.loc[na & cru.str.contains(r"_BOT\b|_BOTT")] = "Bottle"
        out.loc[na & cru.str.contains(r"_XBT\b|_MBT\b")] = "MBT/XBT"
        out.loc[na & cru.str.contains(r"_TSG\b")] = "TSG"

    return out


def summarize_group(df: pd.DataFrame, depth_col: str, std_centers: np.ndarray, tol: float) -> Dict:
    z = to_numeric(df[depth_col])
    z = z[np.isfinite(z.values)]
    zvals = z.values.astype(float)

    out: Dict = {}
    out["rows"] = int(len(df))
    out["valid_depth_rows"] = int(zvals.size)

    if zvals.size == 0:
        out.update({
            "z_min": np.nan, "z_max": np.nan,
            "unique_depths_n": 0,
            "pct_rows_on_stdgrid": np.nan,
            "pct_unique_depths_on_stdgrid": np.nan,
            "median_abs_dist_to_std": np.nan,
            "p95_abs_dist_to_std": np.nan,
            "median_dz": np.nan,
            "unique_depths_sample": ""
        })
        return out

    unique_sorted = np.sort(np.unique(zvals))
    mind_unique = nearest_distance_to_std(unique_sorted, std_centers)
    on_unique = mind_unique <= tol

    mind_rows = nearest_distance_to_std(zvals, std_centers)

    # dz diagnostic
    if unique_sorted.size >= 2:
        median_dz = float(np.nanmedian(np.diff(unique_sorted)))
    else:
        median_dz = np.nan

    preview = unique_sorted[:MAX_UNIQUE_DEPTHS_TO_STORE]
    tail = f"... (+{unique_sorted.size - MAX_UNIQUE_DEPTHS_TO_STORE} more)" if unique_sorted.size > MAX_UNIQUE_DEPTHS_TO_STORE else ""
    unique_preview = " ".join([f"{v:.2f}" for v in preview]) + (f" {tail}" if tail else "")

    out.update({
        "z_min": float(np.nanmin(zvals)),
        "z_max": float(np.nanmax(zvals)),
        "unique_depths_n": int(unique_sorted.size),
        "pct_rows_on_stdgrid": float(np.mean(mind_rows <= tol) * 100.0),
        "pct_unique_depths_on_stdgrid": float(np.mean(on_unique) * 100.0),
        "median_abs_dist_to_std": float(np.nanmedian(mind_rows)),
        "p95_abs_dist_to_std": float(np.nanpercentile(mind_rows, 95)),
        "median_dz": median_dz,
        "unique_depths_sample": unique_preview
    })
    return out


def profile_match_fraction(df: pd.DataFrame, profile_id_col: Optional[str], depth_col: str,
                           std_centers: np.ndarray, tol: float) -> Optional[float]:
    if profile_id_col is None or profile_id_col not in df.columns:
        return None

    pid = df[profile_id_col]
    z = to_numeric(df[depth_col])

    tmp = pd.DataFrame({"pid": pid, "z": z})
    tmp = tmp[np.isfinite(tmp["z"].values)]
    if tmp.empty:
        return np.nan

    def is_profile_on_grid(g: pd.DataFrame) -> bool:
        zv = g["z"].values.astype(float)
        mind = nearest_distance_to_std(zv, std_centers)
        return bool(np.all(mind <= tol))

    flags = tmp.groupby("pid", sort=False).apply(is_profile_on_grid)
    return float(flags.mean() * 100.0)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []
    for p in INPUTS:
        files.extend(list_supported_files(p))

    if not files:
        raise SystemExit("No supported files found. Check INPUTS.")

    summary_rows: List[Dict] = []
    profmatch_rows: List[Dict] = []

    for fp in files:
        # --- read
        try:
            df0 = read_any_table(fp)
        except Exception as e:
            summary_rows.append({"file": str(fp), "status": "READ_FAIL", "error": str(e)})
            continue

        # --- basin filter DJ3
        df, basin_col, n_dj3 = filter_to_basin(df0, TARGET_BASIN)
        if basin_col is None:
            summary_rows.append({
                "file": str(fp),
                "status": "NO_BASIN_COL",
                "error": f"Could not find a basin column containing '{TARGET_BASIN}'.",
            })
            continue
        if n_dj3 == 0:
            summary_rows.append({
                "file": str(fp),
                "status": "NO_ROWS_IN_BASIN",
                "basin_col": basin_col,
                "error": f"No rows for basin '{TARGET_BASIN}'.",
            })
            continue

        # --- detect cols (after basin filtering, same columns anyway)
        try:
            cols = detect_columns(df)
        except Exception as e:
            summary_rows.append({
                "file": str(fp),
                "status": "COL_DETECT_FAIL",
                "basin_col": basin_col,
                "error": str(e),
            })
            continue

        # --- SDN-specific: map to 4 platform classes only + exclude rest
        is_sdn = (SDN_PLATFORM_TEXT_COL in df.columns) and (SDN_DEPTH_COL in df.columns)
        if is_sdn:
            df["platform_class"] = sdn_platform_class_from_instrument(
                df,
                instrument_col=SDN_PLATFORM_TEXT_COL,
                cruise_col="Cruise" if "Cruise" in df.columns else None
            )
            df = df[df["platform_class"].isin(VALID_PLATFORMS)].copy()
            if df.empty:
                summary_rows.append({
                    "file": str(fp),
                    "status": "SDN_ALL_EXCLUDED",
                    "basin_col": basin_col,
                    "error": "After restricting to {TSG,MBT/XBT,Bottle,CTD}, no rows remain for DJ3.",
                })
                continue

            # force SDN columns
            cols = DetectedCols(
                depth=SDN_DEPTH_COL,
                profile_id=(SDN_PROFILE_ID_COL if SDN_PROFILE_ID_COL in df.columns else cols.profile_id),
                platform="platform_class",
                depth_kind="depth",
            )

        # --- grouping: ALL + per-platform (if available)
        group_specs: List[Tuple[str, Optional[str]]] = []
        if SPLIT_BY_PLATFORM_IF_AVAILABLE and cols.platform is not None and cols.platform in df.columns:
            group_specs = [("ALL", None)]
            platforms = np.sort(df[cols.platform].astype(str).fillna("NA").unique())
            # keep sane limit
            platforms = platforms[:200]
            for pval in platforms:
                group_specs.append((str(pval), cols.platform))
        else:
            group_specs = [("ALL", None)]

        for label, plat_col in group_specs:
            if plat_col is None or label == "ALL":
                sub = df
            else:
                sub = df[df[plat_col].astype(str).fillna("NA") == label]

            stats = summarize_group(sub, cols.depth, STD_CENTERS, STD_TOL)

            row = {
                "file": str(fp),
                "status": "OK",
                "target_basin": TARGET_BASIN,
                "basin_col": basin_col,
                "depth_col": cols.depth,
                "depth_kind": cols.depth_kind,
                "profile_id_col": cols.profile_id or "",
                "platform_col": cols.platform or "",
                "platform_value": label,
            }
            row.update(stats)
            summary_rows.append(row)

            if COMPUTE_PROFILE_MATCH:
                pm = profile_match_fraction(sub, cols.profile_id, cols.depth, STD_CENTERS, STD_TOL)
                if pm is not None:
                    profmatch_rows.append({
                        "file": str(fp),
                        "target_basin": TARGET_BASIN,
                        "basin_col": basin_col,
                        "platform_value": label,
                        "profile_id_col": cols.profile_id,
                        "pct_profiles_all_depths_on_stdgrid": pm,
                    })

    summary_df = pd.DataFrame(summary_rows)
    out1 = OUTDIR / "standard_depths_summary_DJ3.csv"
    summary_df.to_csv(out1, index=False)

    print(f"\nWrote: {out1}")
    # Preview: show ALL only if present
    if "platform_value" in summary_df.columns:
        preview = summary_df[summary_df["platform_value"].fillna("") == "ALL"].copy()
    else:
        preview = summary_df.copy()

    show_cols = [
        "file", "status", "target_basin", "basin_col",
        "depth_col", "profile_id_col", "platform_col",
        "rows", "valid_depth_rows", "z_min", "z_max",
        "unique_depths_n", "pct_rows_on_stdgrid", "pct_unique_depths_on_stdgrid",
        "median_abs_dist_to_std", "p95_abs_dist_to_std", "median_dz"
    ]
    show_cols = [c for c in show_cols if c in preview.columns]
    pd.set_option("display.max_colwidth", 140)
    if not preview.empty:
        print("\nPreview (platform_value==ALL):")
        print(preview[show_cols].to_string(index=False))

    if COMPUTE_PROFILE_MATCH and profmatch_rows:
        prof_df = pd.DataFrame(profmatch_rows)
        out2 = OUTDIR / "standard_depths_profile_match_DJ3.csv"
        prof_df.to_csv(out2, index=False)
        print(f"\nWrote: {out2}")
        print("\nProfile-match preview:")
        print(prof_df.head(30).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
