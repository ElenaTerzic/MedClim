#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reads : Data/Seadatanet/*_mainland_special_excluded_no_argo_copies.csv, Data/Argo/CSV/Argo_with_basins_mainland_special_excluded.csv, Data/Adriatic/Processed/*_long_fixed_mainland_special_excluded.csv
# Writes: Stats/MultiSource_Climatology_BINNED_ELIG/<period>/{per_source,POOLED_RAW}/climatology_*.csv, cast_index_*.csv

"""
Multi-source seasonal climatology builder (2-pass Manca-style), using
STD-centered BINNING (no interpolation) and *per-cast bin means*.

This standalone version supports TWO basin modes:
  1) ATOMIC  : original Manca basin codes in the input CSVs (DF1, DJ4, DL2, ...)
  2) COMPOSED: merged “paper regions” (alb, west, nwm, tyr, sic, ion, lev, naeg, saeg, ...)
     defined in COMPOSED_BASINS below (mapping from atomic -> composed).

Key features:
- Basin-specific bin schemes: DEFAULT (Med-wide) and ADRIATIC (DJ1/DJ2/DJ3 and Adriatic composed basins).
- Robust cast/profile identifiers per source (supports "||" alternatives).
- Coverage fractions computed vs eligible casts per bin:
      eligible(bin) = casts whose depth span intersects the bin
                    = (zmin_cast <= bin_high) & (zmax_cast >= bin_low)
      frac_in_bin    = n_in_bin / n_eligible(bin)

Outputs written for EACH period:
  OUTDIR_BASE/<period>/
      per_source/   climatology_<SOURCE>_<period>_<basin_mode>_after_sigma.csv
      POOLED_RAW/   climatology_POOLED_RAW_<period>_<basin_mode>_after_sigma.csv

Columns:
  basin, season, z_m, bin_scheme,
  n_profiles_total, n_eligible_in_bin,
  n_T, T_mean, T_std, frac_T_in_bin, frac_T_in_bin_allcasts, flag_T_lowcov,
  n_S, S_mean, S_std, frac_S_in_bin, frac_S_in_bin_allcasts, flag_S_lowcov

Notes:
- For COMPOSED mode, rows whose atomic basin is not mapped are skipped by default.
- If you want to include “unmapped” basins in COMPOSED mode, set INCLUDE_UNMAPPED_IN_COMPOSED=True.

Author: adapted for Elena’s MedClim pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Iterable

import numpy as np


# Optional: TEOS-10 for sigma0 (PDA)
try:
    import gsw  # type: ignore
    HAS_GSW = True
except Exception:
    HAS_GSW = False
import pandas as pd

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Scripts.paths import DATA, STATS


# =============================================================================
# CONFIG YOU EDIT
# =============================================================================

PERIODS = {
    "20c_1950_1999": (1950, 1999),
    "21c_2000_2025": (2000, 2025), #before I had 2019 for AA
    # Restored for the Ocean Science revision: the OS manuscript states
    # 2000-2019 for Figs 1-2 and its Section 2, so the climatologies behind
    # the revised figures and the steric recalculation have to match that.
    "21c_2000_2019": (2000, 2019),
}

OUTDIR_BASE = STATS / "MultiSource_Climatology_BINNED_ELIG"

SAVE_FILES = True
SAVE_PER_SOURCE = True
SAVE_POOLED_RAW = True
SAVE_PASS1 = False   # usually you only need PASS2 outputs

CHUNKSIZE = 400_000

SEASONS = {
    "JFM": {1, 2, 3},
    "AMJ": {4, 5, 6},
    "JAS": {7, 8, 9},
    "OND": {10, 11, 12},
}

# --- Bin centers (DEFAULT, Mediterranean-wide) ---
STD_CENTERS_DEFAULT = np.array([
    5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200,
    250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,
    1200, 1300, 1400, 1500, 1750, 2000, 2500, 3000,
    3500, 4000, 4500, 5000
], dtype=float)

# --- Bin centers (ADRIATIC DJ1/DJ2/DJ3) ---
# Focused + capped around the Adriatic deep pit (~1200 m).
STD_CENTERS_ADRIATIC = np.array([
    5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200,
    250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200
], dtype=float)

# --- Low-coverage thresholds (eligible-denom) ---
MIN_FRAC_COVERAGE_T = 0.05
MIN_FRAC_COVERAGE_S = 0.05

# PASS 2 sigma rejection multiplier (set to 0 to disable PASS2 masking)
SIGMA_K = 1.0

# Depth limits (meters; dbar treated ~ meters for binning)
ZMIN, ZMAX = 0.0, 6000.0

# QC rules
GOOD_QF_SDN  = {"1", "2"}
GOOD_QF_ARGO = {"1", "2"}

# Optional: write a cast index (for later spatial-bias checks).
EXPORT_CAST_INDEX = True

# =============================================================================
# OPTIONAL RUNTIME OVERRIDES (ENV VARS) - for splitting long runs
# =============================================================================
# You can override run scope without editing the script, e.g.:
#   RUN_BASIN_MODES=atomic RUN_PERIODS=21c_2000_2019 RUN_PER_SOURCE=0 RUN_POOLED=1 \
#       python multi_source_seasonal_climatology_TS_SIG0_COMPOSED_FASTPROD.py
#
# Defaults are the config values above.

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v == "":
        return default
    return v in ("1", "true", "yes", "y", "on")

def _env_list(name: str, default: List[str]) -> List[str]:
    v = os.environ.get(name, "").strip()
    if v == "":
        return default
    return [x.strip() for x in v.split(",") if x.strip()]



# =============================================================================
# COMPOSED BASINS (paper regions)
# =============================================================================

# IMPORTANT:
# - Keys are composed basin IDs (lowercase recommended).
# - Members are atomic basin IDs as they appear in your "basin" columns (usually uppercase like DF1).
# - For COMPOSED mode, an atomic basin not present here is skipped unless INCLUDE_UNMAPPED_IN_COMPOSED=True.
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

# In COMPOSED mode, what to do if an atomic basin is not mapped?
# - False (recommended): skip unmapped atomic basins
# - True: keep them as themselves (their basin ID remains the atomic one)
INCLUDE_UNMAPPED_IN_COMPOSED = False

# Which basin IDs should use the Adriatic bin scheme?
# - Atomic: DJ1 DJ2 DJ3
# - Composed: nadr madr sadr (these are 1:1 with DJ1/2/3 but keep consistent)
ADRIATIC_BASINS = {"DJ1", "DJ2", "DJ3", "NADR", "MADR", "SADR"}


# =============================================================================
# BIN SCHEMES (supports basin-specific bins)
# =============================================================================

@dataclass(frozen=True)
class BinScheme:
    name: str
    centers: np.ndarray
    edges: np.ndarray
    low: np.ndarray
    high: np.ndarray

    @property
    def nbins(self) -> int:
        return int(self.centers.size)


def make_centered_bin_edges(centers: np.ndarray, first_edge: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build contiguous bin edges from centers using midpoints.
    edges has length len(centers)+1.

    We force the first edge to 0.0 by default (gives 0–7.5 m for the first bin if centers start at 5,10,...).
    """
    c = np.asarray(centers, float).copy()
    c.sort()
    if c.size < 2:
        raise ValueError("Need >=2 centers.")
    mid = 0.5 * (c[:-1] + c[1:])
    first = float(first_edge)
    last  = float(c[-1] + 0.5 * (c[-1] - c[-2]))
    edges = np.concatenate([[first], mid, [last]])
    return c, edges


def build_scheme(name: str, centers: np.ndarray) -> BinScheme:
    c, e = make_centered_bin_edges(centers, first_edge=0.0)
    low = e[:-1]
    high = e[1:]
    return BinScheme(name=name, centers=c, edges=e, low=low, high=high)


SCHEME_DEFAULT = build_scheme("DEFAULT", STD_CENTERS_DEFAULT)
SCHEME_ADRIATIC = build_scheme("ADRIATIC", STD_CENTERS_ADRIATIC)


def scheme_for_basin(basin: str) -> BinScheme:
    b = str(basin).strip().upper()
    if b in ADRIATIC_BASINS:
        return SCHEME_ADRIATIC
    return SCHEME_DEFAULT


# =============================================================================
# SOURCE SPECS
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Set

@dataclass(frozen=True)
class SourceSpec:
    # --- required (no defaults) FIRST
    name: str
    path: Path
    time_col: str
    basin_col: str
    depth_col: str
    temp_col: str
    sal_col: str
    cast_id_cols: List[str]

    # --- optional / defaults AFTER
    add_date_to_cast_id: bool = True

    temp_qc_col: Optional[str] = None
    sal_qc_col: Optional[str] = None
    depth_qc_col: Optional[str] = None
    good_qf: Optional[Set[str]] = None

    basin_remap: Optional[Dict[str, str]] = None
    depth_kind: str = "depth_m"

    lon_col: Optional[str] = None
    lat_col: Optional[str] = None

    # keep time_format LAST (it has a default)
    time_format: Optional[str] = None


SOURCES: List[SourceSpec] = [
    SourceSpec(
        name="SDN",
        path=DATA / "Seadatanet" / "Seadatanet.profiles_only_with_basins_mainland_special_excluded_no_argo_copies.csv",
        time_col="yyyy-mm-ddThh:mm:ss.sss",
        basin_col="basin",
        depth_col="Depth [m]",
        lon_col="Longitude [degrees_east]",
        lat_col="Latitude [degrees_north]",
        temp_col="ITS-90 water temperature [degrees C]",
        sal_col="Water body salinity [per mille]",
        cast_id_cols=[
            "LOCAL_CDI_ID||LOCAL_CDI_ID:METAVAR:INDEXED_TEXT",
            "Cruise",
            "Station:METAVAR:INDEXED_TEXT",
        ],
        add_date_to_cast_id=False,
        temp_qc_col="QV:SEADATANET.1",
        sal_qc_col="QV:SEADATANET.2",
        good_qf=GOOD_QF_SDN,
        basin_remap=None,
        depth_kind="depth_m",
    ),
    SourceSpec(
        name="ARGO",
        path=DATA / "Argo" / "CSV" / "Argo_with_basins_mainland_special_excluded.csv",
        time_col="TIME",
        time_format="%Y-%m-%dT%H:%M:%S",
        basin_col="basin",
        depth_col="PRES",  # dbar ~ meters is fine for binning
        temp_col="TEMP",
        sal_col="PSAL",
        cast_id_cols=["PLATFORM_NUMBER", "CYCLE_NUMBER"],
        add_date_to_cast_id=False,
        temp_qc_col="TEMP_QC",
        sal_qc_col="PSAL_QC",
        depth_qc_col="PRES_QC",
        good_qf=GOOD_QF_ARGO,
        basin_remap=None,
        depth_kind="pres_dbar",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
    ),
    # --- Adriatic local datasets (fixed) ---
    SourceSpec(
        name="CTD_HR_2019_2023_MFSD",
        path=DATA / "Adriatic" / "Processed" / "CTD_HR_2019_2023_MFSD_long_fixed_mainland_special_excluded.csv",
        time_col="TIME",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        cast_id_cols=["STATION"],
        add_date_to_cast_id=True,
        basin_remap=None,
        depth_kind="depth_m",
    ),
    SourceSpec(
        name="RB_Northern_Adriatic_Transect_1979_2018",
        path=DATA / "Adriatic" / "Processed" / "RB_Northern_Adriatic_Transect_1979_2018_long_fixed_mainland_special_excluded.csv",
        time_col="TIME",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        cast_id_cols=["STATION"],
        add_date_to_cast_id=True,
        basin_remap=None,
        depth_kind="depth_m",
    ),
    SourceSpec(
        name="Palagruza_Transect_1952_2018",
        path=DATA / "Adriatic" / "Processed" / "Palagruza_Transect_1952_2018_long_fixed_mainland_special_excluded.csv",
        time_col="TIME",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        cast_id_cols=["STATION"],
        add_date_to_cast_id=True,
        basin_remap=None,
        depth_kind="depth_m",
    ),
    SourceSpec(
        name="IOR_Adriatic_database_1987_2019",
        path=DATA / "Adriatic" / "Processed" / "IOR_Adriatic_database_1987_2019_long_fixed_mainland_special_excluded.csv",
        time_col="TIME",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        cast_id_cols=["STATION", "CAST_IDX"],
        add_date_to_cast_id=True,
        basin_remap=None,
        depth_kind="depth_m",
    ),
]


# =============================================================================
# UTILS
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
    return s


def apply_basin_remap(basin: str, remap: Optional[Dict[str, str]]) -> str:
    if remap is None:
        return basin
    key = basin.strip().lower()
    return remap.get(key, basin)


def depth_to_meters(z: np.ndarray, kind: str) -> np.ndarray:
    # Treat dbar ~ meters (fine for binning)
    return np.asarray(z, float)


def _qc_mask(g: pd.DataFrame, qc_col: Optional[str], good_qf: Optional[set]) -> np.ndarray:
    if qc_col is None or good_qf is None or qc_col not in g.columns:
        return np.ones(len(g), dtype=bool)
    q = g[qc_col].astype(str).str.strip().to_numpy()
    return np.isin(q, list(good_qf))


def sanitize_values(v: np.ndarray, kind: str) -> np.ndarray:
    v = np.asarray(v, float)
    v[np.abs(v) > 1e4] = np.nan
    if kind == "TEMP":
        v[(v < -5.0) | (v > 45.0)] = np.nan
    elif kind == "SAL":
        v[(v < 0.0) | (v > 60.0)] = np.nan
    return v


def _resolve_col_choice(chunk_cols: List[str], spec_name: str, col_spec: str) -> str:
    """
    col_spec may include alternatives separated by "||". Return the first that exists in chunk_cols.
    """
    alts = [c.strip() for c in str(col_spec).split("||") if c.strip()]
    for c in alts:
        if c in chunk_cols:
            return c
    preview = ", ".join(chunk_cols[:50])
    raise ValueError(
        f"[{spec_name}] None of these cast-id columns exist in the CSV: {alts}\n"
        f"First 50 columns in file: {preview}"
    )


def build_atomic_to_composed_map(comp: Dict[str, Dict[str, object]]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for comp_id, info in comp.items():
        members = info.get("members", [])
        for b in members:
            m[str(b).strip().upper()] = str(comp_id).strip()
    return m


ATOMIC_TO_COMPOSED = build_atomic_to_composed_map(COMPOSED_BASINS)


def map_basin_id(basin_atomic: str, basin_mode: str) -> str:
    """
    basin_mode:
      - "atomic": return atomic basin ID (uppercase)
      - "composed": return composed basin ID (lowercase keys), or "" if unmapped and INCLUDE_UNMAPPED_IN_COMPOSED=False
    """
    b = str(basin_atomic).strip().upper()
    if basin_mode == "atomic":
        return b

    if basin_mode != "composed":
        raise ValueError(f"Invalid basin_mode={basin_mode!r}. Use 'atomic' or 'composed'.")

    comp = ATOMIC_TO_COMPOSED.get(b)
    if comp is None:
        return b if INCLUDE_UNMAPPED_IN_COMPOSED else ""
    return comp


# Helpers to detect lat and lon data

LON_CANDIDATES = [
    "Longitude [degrees_east]",
    "Longitude [degrees_east]:METAVAR:FLOAT",
    "Longitude [degrees_east]:METAVAR:INDEXED_TEXT",
    "lon", "longitude", "LONGITUDE", "LONG", "x", "X"
]
LAT_CANDIDATES = [
    "Latitude [degrees_north]",
    "Latitude [degrees_north]:METAVAR:FLOAT",
    "Latitude [degrees_north]:METAVAR:INDEXED_TEXT",
    "lat", "latitude", "LATITUDE", "LAT", "y", "Y"
]
TIME_CANDIDATES = [
    "yyyy-mm-ddThh:mm:ss.sss",
    "yyyy-mm-ddthh:mm:ss.sss",
    "time", "date_time", "datetime", "DATE_TIME", "ISO_DATETIME_T", "TIME"
]

def _first_valid_numeric(series: pd.Series):
    """Return first non-NaN numeric value; handles comma decimals."""
    if series is None:
        return np.nan
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    # handle comma decimals (e.g. '14,32')
    s = s.str.replace(",", ".", regex=False)
    v = pd.to_numeric(s, errors="coerce")
    v = v.dropna()
    return float(v.iloc[0]) if len(v) else np.nan

def _first_valid_time(series: pd.Series):
    if series is None:
        return pd.NaT
    t = pd.to_datetime(series, errors="coerce")
    t = t.dropna()
    return t.iloc[0] if len(t) else pd.NaT

def pick_col(df: pd.DataFrame, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def extract_lon_lat_time_from_cast(g: pd.DataFrame):
    """Return (lon, lat, time) for a cast/group, using robust column detection.

    Note: time is parsed from the FIRST value only (scalar parse) to avoid slow
    per-element parsing warnings on long Series.
    """
    lon_col = pick_col(g, LON_CANDIDATES)
    lat_col = pick_col(g, LAT_CANDIDATES)
    tim_col = pick_col(g, TIME_CANDIDATES)

    lon = _first_valid_numeric(g[lon_col]) if lon_col else np.nan
    lat = _first_valid_numeric(g[lat_col]) if lat_col else np.nan
    tim = pd.to_datetime(g[tim_col].iloc[0], errors="coerce") if tim_col else pd.NaT
    return lon, lat, tim



# =============================================================================
# STREAMING STATS (WELFORD)
# =============================================================================

class Welford:
    __slots__ = ("n", "mean", "M2")

    def __init__(self, size: int):
        self.n = np.zeros(size, dtype=np.int64)
        self.mean = np.zeros(size, dtype=float)
        self.M2 = np.zeros(size, dtype=float)

    def update_vec(self, x: np.ndarray):
        x = np.asarray(x, float)
        m = np.isfinite(x)
        if not np.any(m):
            return
        xm = x[m]
        n_old = self.n[m]
        self.n[m] = n_old + 1
        delta = xm - self.mean[m]
        self.mean[m] += delta / self.n[m]
        delta2 = xm - self.mean[m]
        self.M2[m] += delta * delta2

    def finalize(self):
        var = np.full_like(self.mean, np.nan, dtype=float)
        ok = self.n > 1
        var[ok] = self.M2[ok] / (self.n[ok] - 1)
        std = np.sqrt(var)
        mean = self.mean.copy()
        mean[self.n == 0] = np.nan
        return mean, std, self.n.copy()


# =============================================================================
# STREAMING PROFILES (FILE READER) - robust cast id
# =============================================================================

def iter_profiles_csv(spec: SourceSpec, chunksize: int = CHUNKSIZE) -> Iterable[pd.DataFrame]:
    """
    Stream per-cast/profile DataFrames from a large CSV.

    Speed notes:
    - Uses usecols to avoid loading unused columns.
    - Converts numeric columns (depth/temp/sal/lon/lat) once per chunk (not per-cast).
    - Parses datetime for the whole chunk ONLY when add_date_to_cast_id=True (needed for cast-id suffix).
    """
    if not spec.path.exists():
        raise FileNotFoundError(f"Missing input: {spec.path}")

    # --- Determine minimal columns to read
    usecols = {
        spec.time_col, spec.basin_col, spec.depth_col, spec.temp_col, spec.sal_col
    }

    # Cast-id columns may include alternatives separated by "||"
    for item in spec.cast_id_cols:
        for alt in str(item).split("||"):
            alt = alt.strip()
            if alt:
                usecols.add(alt)

    # Optional QC columns
    if spec.temp_qc_col:
        usecols.add(spec.temp_qc_col)
    if spec.sal_qc_col:
        usecols.add(spec.sal_qc_col)
    if spec.depth_qc_col:
        usecols.add(spec.depth_qc_col)

    # Optional spatial columns (needed for cast_index and SIG0)
    if spec.lon_col:
        for alt in str(spec.lon_col).split("||"):
            alt = alt.strip()
            if alt:
                usecols.add(alt)
    if spec.lat_col:
        for alt in str(spec.lat_col).split("||"):
            alt = alt.strip()
            if alt:
                usecols.add(alt)
    def _usecols(c: str) -> bool:
        if c in usecols:
            return True
        # SDN sometimes has lon/lat columns with extra suffixes; keep them
        if spec.name == "SDN":
            cl = c.lower()
            if cl.startswith("longitude") or "degrees_east" in cl:
                return True
            if cl.startswith("latitude") or "degrees_north" in cl:
                return True
        return False

    reader = pd.read_csv(
        spec.path,
        chunksize=chunksize,
        dtype=str,
        low_memory=False,
        usecols=_usecols
    )

    carry = None

    for chunk in reader:
        if carry is not None and len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)

        # Convert main numeric columns ONCE per chunk
        for col in (spec.depth_col, spec.temp_col, spec.sal_col):
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        # Convert lon/lat if present (comma-decimal tolerant). For SDN we may have multiple variants.
        lonlat_cols = []
        # explicit cols
        for col in (spec.lon_col, spec.lat_col):
            if col and (col in chunk.columns):
                lonlat_cols.append(col)
        # SDN variants
        if spec.name == "SDN":
            for c in chunk.columns:
                cl = c.lower()
                if cl.startswith("longitude") or "degrees_east" in cl:
                    lonlat_cols.append(c)
                if cl.startswith("latitude") or "degrees_north" in cl:
                    lonlat_cols.append(c)
        for col in sorted(set(lonlat_cols)):
            s = chunk[col].astype(str).str.strip().str.replace(",", ".", regex=False)
            chunk[col] = pd.to_numeric(s, errors="coerce")

        # Parse time only if we need a date suffix in the cast-id
        t = None
        if spec.add_date_to_cast_id:
            if spec.time_format:
                t = pd.to_datetime(chunk[spec.time_col], format=spec.time_format, errors="coerce")
            else:
                t = pd.to_datetime(chunk[spec.time_col], errors="coerce")

        # Resolve cast-id columns (supports "||" alternatives)
        chosen_cols: List[str] = []
        for item in spec.cast_id_cols:
            chosen_cols.append(_resolve_col_choice(list(chunk.columns), spec.name, item))

        parts = [chunk[c].astype(str) for c in chosen_cols]

        # Optional date suffix
        if spec.add_date_to_cast_id:
            date_str = t.dt.strftime("%Y-%m-%d").fillna("NaT") if t is not None else pd.Series(["NaT"] * len(chunk))
            parts.append(date_str.astype(str))

        pid = parts[0]
        for p in parts[1:]:
            pid = pid + "|" + p

        # Prefix by source name so pooled processing can never collide across sources
        chunk["_profile"] = spec.name + "|" + pid

        # carry-over rows that share the last profile id in the chunk
        last_prof = chunk["_profile"].iloc[-1]
        carry = chunk[chunk["_profile"] == last_prof].copy()
        chunk = chunk[chunk["_profile"] != last_prof]
        if len(chunk) == 0:
            continue

        for _, g in chunk.groupby("_profile", sort=False):
            yield g

    if carry is not None and len(carry):
        for _, g in carry.groupby("_profile", sort=False):
            yield g



# =============================================================================
# BINNING CORE (per-cast bin means) - basin-specific scheme
# =============================================================================

def profile_bin_means(z: np.ndarray, v: np.ndarray, scheme: BinScheme) -> np.ndarray:
    """
    Given arrays (z,v) already QC-filtered, compute per-bin MEANS for this cast.
    Returns vector length scheme.nbins with NaN where bin has no samples.
    """
    out = np.full(scheme.nbins, np.nan, dtype=float)
    if z.size == 0:
        return out

    bi = np.searchsorted(scheme.edges, z, side="right") - 1
    ok = (bi >= 0) & (bi < scheme.nbins) & np.isfinite(v)
    bi = bi[ok]
    vv = v[ok]
    if bi.size == 0:
        return out

    cnt = np.bincount(bi, minlength=scheme.nbins).astype(float)
    s   = np.bincount(bi, weights=vv, minlength=scheme.nbins).astype(float)
    m = cnt > 0
    out[m] = s[m] / cnt[m]
    return out


def collect_cast_binned(
    g: pd.DataFrame,
    spec: SourceSpec,
    scheme: BinScheme
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, float, float]:

    z_raw = pd.to_numeric(g.get(spec.depth_col), errors="coerce").to_numpy()
    z = depth_to_meters(z_raw, spec.depth_kind)

    mZ = np.isfinite(z) & (z >= ZMIN) & (z <= ZMAX)
    mZ &= _qc_mask(g, spec.depth_qc_col, spec.good_qf)

    if not np.any(mZ):
        return (
            np.full(scheme.nbins, np.nan),
            np.full(scheme.nbins, np.nan),
            np.full(scheme.nbins, np.nan),
            False,
            np.nan,
            np.nan,
        )

    z_valid = z[mZ]
    zmin_cast = float(np.nanmin(z_valid))
    zmax_cast = float(np.nanmax(z_valid))

    T = pd.to_numeric(g.get(spec.temp_col), errors="coerce").to_numpy()
    S = pd.to_numeric(g.get(spec.sal_col), errors="coerce").to_numpy()

    mT = mZ & np.isfinite(T) & (T >= -5.0) & (T <= 45.0)
    mT &= _qc_mask(g, spec.temp_qc_col, spec.good_qf)

    mS = mZ & np.isfinite(S) & (S >= 0.0) & (S <= 60.0)
    mS &= _qc_mask(g, spec.sal_qc_col, spec.good_qf)

    Tbin = np.full(scheme.nbins, np.nan)
    Sbin = np.full(scheme.nbins, np.nan)
    SIG0bin = np.full(scheme.nbins, np.nan)

    if np.any(mT):
        Tbin = sanitize_values(profile_bin_means(z[mT], T[mT], scheme), "TEMP")

    if np.any(mS):
        Sbin = sanitize_values(profile_bin_means(z[mS], S[mS], scheme), "SAL")

    # ---------------------------
    # SIGMA0 (PDA)
    # ---------------------------
    if HAS_GSW and np.any(mT & mS):

        mSIG = mT & mS
        p = z  # dbar ~ m for Mediterranean is acceptable

        # --- Robust lon/lat extraction
        lon, lat, _ = extract_lon_lat_time_from_cast(g)

        if not np.isfinite(lon):
            lon = 0.0
        if not np.isfinite(lat):
            lat = 0.0

        try:
            npts = np.count_nonzero(mSIG)
            lon_vec = np.full(npts, lon, dtype=float)
            lat_vec = np.full(npts, lat, dtype=float)

            SA = gsw.SA_from_SP(S[mSIG], p[mSIG], lon_vec, lat_vec)
            CT = gsw.CT_from_t(SA, T[mSIG], p[mSIG])
            sigma0 = gsw.sigma0(SA, CT)

            SIG0 = np.full_like(T, np.nan, dtype=float)
            SIG0[mSIG] = sigma0

            SIG0bin = profile_bin_means(z[mSIG], SIG0[mSIG], scheme)

        except Exception:
            pass

    return Tbin, Sbin, SIG0bin, True, zmin_cast, zmax_cast


# =============================================================================
# PASS 2 SUPPORT
# =============================================================================

def build_pass2_dicts(accT, accS):
    means: Dict[Tuple[str, str, str], np.ndarray] = {}
    stds:  Dict[Tuple[str, str, str], np.ndarray] = {}

    for (basin, season), w in accT.items():
        mu, sd, _ = w.finalize()
        means[("T", basin, season)] = mu
        stds[("T", basin, season)] = sd

    for (basin, season), w in accS.items():
        mu, sd, _ = w.finalize()
        means[("S", basin, season)] = mu
        stds[("S", basin, season)] = sd

    return means, stds


# =============================================================================
# MAIN COMPUTATION: PASS1 / PASS2
# =============================================================================

def pass_stats_for_sources(
    sources: List[SourceSpec],
    year_min: int,
    year_max: int,
    basin_mode: str,
    first_pass_means: Optional[Dict[Tuple[str, str, str], np.ndarray]] = None,
    first_pass_stds: Optional[Dict[Tuple[str, str, str], np.ndarray]] = None,
):
    """
    Returns:
      accT[(basin, season)] = Welford(nbins_for_basin)
      accS[(basin, season)] = Welford(nbins_for_basin)
      accSIG0[(basin, season)] = Welford(nbins_for_basin)
      rd(nbins_for_basin)
      n_total[(basin, season)] = number of casts in group
      n_eligible[(basin, season)] = array (nbins_for_basin,) counts of eligible casts per bin
      scheme_used[(basin, season)] = BinScheme
      cast_index_df (optional) if EXPORT_CAST_INDEX is True
    """
    accT: Dict[Tuple[str, str], Welford] = {}
    accS: Dict[Tuple[str, str], Welford] = {}
    accSIG0: Dict[Tuple[str, str], Welford] = {}
    n_total: Dict[Tuple[str, str], int] = {}
    n_eligible: Dict[Tuple[str, str], np.ndarray] = {}
    scheme_used: Dict[Tuple[str, str], BinScheme] = {}

    cast_rows = [] if EXPORT_CAST_INDEX else None

    for spec in sources:
        for g in iter_profiles_csv(spec, chunksize=CHUNKSIZE):

            # basin read
            basin_raw = g[spec.basin_col].iloc[0] if spec.basin_col in g.columns else ""
            basin = normalize_basin_value(basin_raw)
            basin = apply_basin_remap(basin, spec.basin_remap)
            if basin == "" or basin.lower() in ("none", "nan"):
                continue

            # map to atomic/composed
            basin_mapped = map_basin_id(basin, basin_mode=basin_mode)
            if basin_mapped == "":
                continue  # unmapped in composed mode

            # time
            t0 = pd.to_datetime(g[spec.time_col].iloc[0], errors="coerce")
            if pd.isna(t0):
                continue
            y = int(t0.year)
            if y < year_min or y > year_max:
                continue
            season = month_to_season(int(t0.month))
            if season is None:
                continue

            key = (basin_mapped, season)
            scheme = scheme_for_basin(basin_mapped)

            # per-cast binned vectors + depth span
            Tbin, Sbin, SIG0bin, has_any_depth, zmin_cast, zmax_cast = collect_cast_binned(g, spec, scheme)
            if not has_any_depth:
                continue

            # init scheme tracking
            if key in scheme_used:
                if scheme_used[key].name != scheme.name or scheme_used[key].nbins != scheme.nbins:
                    raise RuntimeError(f"Bin scheme mismatch within group {key}: {scheme_used[key]} vs {scheme}")
            else:
                scheme_used[key] = scheme

            # denominator: all casts that pass basin+time+depth sanity
            n_total[key] = n_total.get(key, 0) + 1

            # eligible per bin: cast depth span intersects bin range
            # (optional clarity cap to scheme range)
            zmin_eff = zmin_cast
            zmax_eff = min(zmax_cast, float(scheme.high[-1]))
            elig = (zmin_eff <= scheme.high) & (zmax_eff >= scheme.low)

            if key not in n_eligible:
                n_eligible[key] = np.zeros(scheme.nbins, dtype=np.int64)
            n_eligible[key][elig] += 1

            # Optional cast index export
            if EXPORT_CAST_INDEX and (cast_rows is not None):

                lon, lat, _ = extract_lon_lat_time_from_cast(g)

                cast_rows.append({
                    "source": spec.name,
                    "basin": basin_mapped,
                    "season": season,
                    "time": str(t0),
                    "year": y,
                    "lon": float(lon) if np.isfinite(lon) else np.nan,
                    "lat": float(lat) if np.isfinite(lat) else np.nan,
                    "zmin_cast": float(zmin_cast),
                    "zmax_cast": float(zmax_cast),
                    "bin_scheme": scheme.name,
                    "basin_mode": basin_mode,
                })


            # PASS2 rejection (bin-wise)
            if (first_pass_means is not None) and (first_pass_stds is not None) and (SIGMA_K > 0):
                muT = first_pass_means.get(("T", basin_mapped, season))
                sdT = first_pass_stds.get(("T", basin_mapped, season))
                if (muT is not None) and (sdT is not None):
                    ok = np.isfinite(Tbin) & np.isfinite(muT) & np.isfinite(sdT) & (sdT > 0)
                    Tbin[ok & (np.abs(Tbin - muT) > (SIGMA_K * sdT))] = np.nan
                    Tbin = sanitize_values(Tbin, "TEMP")

                muS = first_pass_means.get(("S", basin_mapped, season))
                sdS = first_pass_stds.get(("S", basin_mapped, season))
                if (muS is not None) and (sdS is not None):
                    ok = np.isfinite(Sbin) & np.isfinite(muS) & np.isfinite(sdS) & (sdS > 0)
                    Sbin[ok & (np.abs(Sbin - muS) > (SIGMA_K * sdS))] = np.nan
                    Sbin = sanitize_values(Sbin, "SAL")

            # Update group stats
            if key not in accT:
                accT[key] = Welford(scheme.nbins)
                accS[key] = Welford(scheme.nbins)
                accSIG0[key] = Welford(scheme.nbins)
            accT[key].update_vec(Tbin)
            accS[key].update_vec(Sbin)
            accSIG0[key].update_vec(SIG0bin)

    cast_index_df = pd.DataFrame(cast_rows) if (EXPORT_CAST_INDEX and cast_rows is not None) else None
    return accT, accS, accSIG0, n_total, n_eligible, scheme_used, cast_index_df


# =============================================================================
# OUTPUT
# =============================================================================

def dump_csv(accT, accS, n_total, n_eligible, scheme_used, out_csv: Path, accSIG0=None):
    rows = []
    keys = sorted(set(list(accT.keys()) + list(accS.keys()) + list(n_total.keys()) + list(n_eligible.keys()) + list(scheme_used.keys())))

    for basin, season in keys:
        scheme = scheme_used.get((basin, season), scheme_for_basin(basin))
        nb = scheme.nbins

        tmean = np.full(nb, np.nan)
        tstd  = np.full(nb, np.nan)
        tn    = np.zeros(nb, int)

        smean = np.full(nb, np.nan)
        sstd  = np.full(nb, np.nan)
        sn    = np.zeros(nb, int)

        sig0_mean = np.full(nb, np.nan)
        sig0_std  = np.full(nb, np.nan)
        sig0_n    = np.zeros(nb, int)

        if (basin, season) in accT:
            tmean, tstd, tn = accT[(basin, season)].finalize()
        if (basin, season) in accS:
            smean, sstd, sn = accS[(basin, season)].finalize()
        if (accSIG0 is not None) and ((basin, season) in accSIG0):
            sig0_mean, sig0_std, sig0_n = accSIG0[(basin, season)].finalize()

        nprof = int(n_total.get((basin, season), 0))
        elig = n_eligible.get((basin, season), np.zeros(nb, dtype=np.int64)).astype(float)

        # Fractions vs eligible casts
        fracT = np.full(nb, np.nan, dtype=float)
        fracS = np.full(nb, np.nan, dtype=float)
        okE = elig > 0
        fracT[okE] = tn[okE] / elig[okE]
        fracS[okE] = sn[okE] / elig[okE]

        # Old denom (all casts) fractions for debugging
        if nprof > 0:
            fracT_all = tn / float(nprof)
            fracS_all = sn / float(nprof)
        else:
            fracT_all = np.full(nb, np.nan, dtype=float)
            fracS_all = np.full(nb, np.nan, dtype=float)

        flagT = (fracT < MIN_FRAC_COVERAGE_T)
        flagS = (fracS < MIN_FRAC_COVERAGE_S)

        for k, zc in enumerate(scheme.centers):
            rows.append({
                "basin": basin,
                "season": season,
                "z_m": float(zc),
                "bin_scheme": scheme.name,

                "n_profiles_total": int(nprof),
                "n_eligible_in_bin": int(elig[k]) if np.isfinite(elig[k]) else 0,

                "n_T": int(tn[k]),
                "T_mean": float(tmean[k]) if np.isfinite(tmean[k]) else np.nan,
                "T_std": float(tstd[k]) if np.isfinite(tstd[k]) else np.nan,
                "frac_T_in_bin": float(fracT[k]) if np.isfinite(fracT[k]) else np.nan,
                "frac_T_in_bin_allcasts": float(fracT_all[k]) if np.isfinite(fracT_all[k]) else np.nan,
                "flag_T_lowcov": int(bool(flagT[k])),

                "n_S": int(sn[k]),
                "S_mean": float(smean[k]) if np.isfinite(smean[k]) else np.nan,
                "S_std": float(sstd[k]) if np.isfinite(sstd[k]) else np.nan,
                "n_SIG0": int(sig0_n[k]),
                "SIG0_mean": float(sig0_mean[k]) if np.isfinite(sig0_mean[k]) else np.nan,
                "SIG0_std": float(sig0_std[k]) if np.isfinite(sig0_std[k]) else np.nan,

                "frac_S_in_bin": float(fracS[k]) if np.isfinite(fracS[k]) else np.nan,
                "frac_S_in_bin_allcasts": float(fracS_all[k]) if np.isfinite(fracS_all[k]) else np.nan,
                "flag_S_lowcov": int(bool(flagS[k])),
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)
    return df


# =============================================================================
# DRIVER
# =============================================================================

def run_one_period(period_label: str, ymin: int, ymax: int, basin_mode: str):
    """
    Run per-source and pooled climatologies for a given period and basin_mode.
    """
    print("=" * 90)
    print(f"PERIOD {period_label}: {ymin}–{ymax} | basin_mode={basin_mode}")

    out_period = OUTDIR_BASE / period_label
    out_per_source = out_period / "per_source"
    out_pooled = out_period / "POOLED_RAW"

    suffix = f"_{basin_mode}"

    # -------------------------
    # Per-source pass1 + pass2
    # -------------------------
    if SAVE_PER_SOURCE:
        for spec in SOURCES:
            print("\n" + "-" * 90)
            print(f"[{spec.name}] PASS 1 ...")
            accT1, accS1, accSIG01, n1, elig1, scheme1, cast_index1 = pass_stats_for_sources(
                [spec], ymin, ymax, basin_mode=basin_mode
            )

            if SAVE_FILES and SAVE_PASS1:
                dump_csv(
                    accT1, accS1, n1, elig1, scheme1,
                    out_per_source / f"climatology_{spec.name}_{period_label}{suffix}_pass1.csv", accSIG0=accSIG01
                )

            means, stds = build_pass2_dicts(accT1, accS1)

            print(f"[{spec.name}] PASS 2 (sigma rejection) ...")
            accT2, accS2, accSIG02, n2, elig2, scheme2, cast_index2 = pass_stats_for_sources(
                [spec], ymin, ymax, basin_mode=basin_mode,
                first_pass_means=means, first_pass_stds=stds
            )

            if SAVE_FILES:
                dump_csv(
                    accT2, accS2, n2, elig2, scheme2,
                    out_per_source / f"climatology_{spec.name}_{period_label}{suffix}_after_sigma.csv", accSIG0=accSIG02
                )

            if SAVE_FILES and EXPORT_CAST_INDEX and (cast_index2 is not None):
                p = out_per_source / f"cast_index_{spec.name}_{period_label}{suffix}.csv"
                p.parent.mkdir(parents=True, exist_ok=True)
                cast_index2.to_csv(p, index=False)
                print("Wrote:", p)

    # -------------------------
    # Pooled RAW pass1 + pass2
    # -------------------------
    if SAVE_POOLED_RAW:
        print("\n" + "-" * 90)
        print("[POOLED_RAW] PASS 1 across ALL enabled sources ...")
        accT1p, accS1p, accSIG01p, n1p, elig1p, scheme1p, cast_index1p = pass_stats_for_sources(
            SOURCES, ymin, ymax, basin_mode=basin_mode
        )

        if SAVE_FILES and SAVE_PASS1:
            dump_csv(
                accT1p, accS1p, n1p, elig1p, scheme1p,
                out_pooled / f"climatology_POOLED_RAW_{period_label}{suffix}_pass1.csv", accSIG0=accSIG01p
            )

        means_p, stds_p = build_pass2_dicts(accT1p, accS1p)

        print("[POOLED_RAW] PASS 2 across ALL enabled sources (sigma rejection) ...")
        accT2p, accS2p, accSIG02p, n2p, elig2p, scheme2p, cast_index2p = pass_stats_for_sources(
            SOURCES, ymin, ymax, basin_mode=basin_mode,
            first_pass_means=means_p, first_pass_stds=stds_p
        )

        if SAVE_FILES:
            outpath = out_pooled / f"climatology_POOLED_RAW_{period_label}{suffix}_after_sigma.csv"
            ncasts_total = sum(n2p.values()) if isinstance(n2p, dict) else 0
            if ncasts_total == 0:
                print(f"[POOLED_RAW] No casts in {period_label} -> skipping {outpath.name}")
            else:
                dump_csv(accT2p, accS2p, n2p, elig2p, scheme2p, outpath, accSIG0=accSIG02p)

        if SAVE_FILES and EXPORT_CAST_INDEX and (cast_index2p is not None):
            p = out_pooled / f"cast_index_POOLED_RAW_{period_label}{suffix}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            cast_index2p.to_csv(p, index=False)
            print("Wrote:", p)


def main():
    print("\nMulti-source seasonal climatology (BINNED, 2-pass, ELIGIBLE-DENOM)")
    print("OUTDIR_BASE:", OUTDIR_BASE)
    print("Sources:", ", ".join([s.name for s in SOURCES]))
    print("Periods:", ", ".join([f"{k}({v[0]}–{v[1]})" for k, v in PERIODS.items()]))
    print("Bin schemes: DEFAULT=%d bins, ADRIATIC=%d bins" % (SCHEME_DEFAULT.nbins, SCHEME_ADRIATIC.nbins))
    print(f"Low-cov thresholds (eligible-denom): T<{MIN_FRAC_COVERAGE_T:.3f}, S<{MIN_FRAC_COVERAGE_S:.3f}")
    print(f"PASS2 rejection: |x-mu| > {SIGMA_K:.1f}σ (set SIGMA_K=0 to disable)")
    print(f"COMPOSED basins: {len(COMPOSED_BASINS)} groups | include_unmapped={INCLUDE_UNMAPPED_IN_COMPOSED}\n")

    # Run scope (can be overridden by env vars)
    global SAVE_PER_SOURCE, SAVE_POOLED_RAW
    SAVE_PER_SOURCE = _env_bool("RUN_PER_SOURCE", SAVE_PER_SOURCE)
    SAVE_POOLED_RAW = _env_bool("RUN_POOLED", SAVE_POOLED_RAW)

    basin_modes = _env_list("RUN_BASIN_MODES", ["atomic", "composed"])
    periods_to_run = _env_list("RUN_PERIODS", list(PERIODS.keys()))

    # Validate requested periods
    for p in periods_to_run:
        if p not in PERIODS:
            raise ValueError(f"RUN_PERIODS contains unknown period: {p}. Known: {list(PERIODS.keys())}")

    for basin_mode in basin_modes:
        print("\n" + "#" * 90)
        print(f"RUN basin_mode={basin_mode} | per_source={SAVE_PER_SOURCE} | pooled_raw={SAVE_POOLED_RAW}")
        print("#" * 90 + "\n")

        for period_label in periods_to_run:
            ymin, ymax = PERIODS[period_label]
            run_one_period(period_label, ymin, ymax, basin_mode=basin_mode)
print("\nDone.")


if __name__ == "__main__":
    main()
