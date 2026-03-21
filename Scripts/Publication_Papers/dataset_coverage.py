#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 1. Observational data set and coverage

LAYOUT
------
Row 1: map across full width
Row 2: temporal stacked bar chart across full width
Row 3: two Hovmöllers side by side

Panels
------
a) Spatial distribution of observations
b) Temporal evolution of observations (unique profiles/casts per year)
c) Whole Mediterranean temperature anomaly
d) NWM temperature anomaly

Important
---------
- Panel (a) uses the working profile_diagnostics files for map points
- Panel (b) uses the exact source definitions and cast/profile logic from
  multi_source_seasonal_climatology.py, so counts are unique profiles per year
  rather than raw rows
- Panels (c-d) keep the yearly anomaly logic based on monthly composed series
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# Optional map libs
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False

try:
    import cmocean
    HAVE_CMOCEAN = True
except Exception:
    HAVE_CMOCEAN = False


# ==========================================================
# IMPORT THE CLIMATOLOGY SOURCE CONFIG
# ==========================================================

import sys
from pathlib import Path

HERE = Path(__file__).resolve()

PROJECT_ROOT = None
for p in [HERE.parent] + list(HERE.parents):
    if (p / "Scripts" / "MultiPlatform" / "core" / "multi_source_seasonal_climatology.py").exists():
        PROJECT_ROOT = p
        break

if PROJECT_ROOT is None:
    raise ImportError(
        "Could not locate MedClim project root containing "
        "Scripts/MultiPlatform/core/multi_source_seasonal_climatology.py"
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from Scripts.MultiPlatform.core import multi_source_seasonal_climatology as msc
except ImportError as e:
    raise ImportError(
        f"Found project root at {PROJECT_ROOT}, but still could not import "
        "Scripts.MultiPlatform.core.multi_source_seasonal_climatology"
    ) from e

# ==========================================================
# USER SETTINGS
# ==========================================================

# ----------------------------
# Map point files
# ----------------------------
SDN_MAP_CSV = "/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins_mainland_special_excluded_profile_diagnostics.csv"
ARGO_MAP_CSV = "/Users/elena/Projects/IRB/MedClim/Data/Argo/CSV/Argo_with_basins_mainland_special_excluded_profile_diagnostics.csv"

ADRIATIC_MAP_CSVS = [
    "/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/CTD_HR_2019_2023_MFSD_long_fixed_mainland_special_excluded_profile_diagnostics.csv",
    "/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/RB_Northern_Adriatic_Transect_1979_2018_long_fixed_mainland_special_excluded_profile_diagnostics.csv",
    "/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/Palagruza_Transect_1952_2018_long_fixed_mainland_special_excluded_profile_diagnostics.csv",
    "/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/IOR_Adriatic_database_1987_2019_long_fixed_mainland_special_excluded_profile_diagnostics.csv",
]

# ----------------------------
# Monthly composed series for Hovmöllers
# ----------------------------
MONTHLY_COMPOSED_CSV = "/Users/elena/Projects/IRB/MedClim/Results/Trends/monthly_timeseries_all_bins_composed.csv"

# ----------------------------
# Output
# ----------------------------
OUTDIR = "/Users/elena/Projects/IRB/MedClim/Plots/Paper_Figures"
OUTPNG = os.path.join(OUTDIR, "Figure1_dataset_coverage_3row.png")
OUTPDF = os.path.join(OUTDIR, "Figure1_dataset_coverage_3row.pdf")
DPI = 400

# ----------------------------
# Time settings
# ----------------------------
YEAR_MIN = 1950
YEAR_MAX = 2025
REF_YEAR_MIN = 1950
REF_YEAR_MAX = 1999

# ----------------------------
# Hovmöller settings
# ----------------------------
HOV_VAR = "T"
REPRESENTATIVE_BASIN = "NWM"
MAX_DEPTH_M = 2000.0
MIN_VALID_MONTHS_PER_YEAR = 1
MIN_N_PER_YEAR = 1.0
APPLY_ROLLING = True
ROLLING_YEARS = 3
ROLLING_CENTER = True
MIN_VALID_YEARS_FOR_ZMAX = 5

# ----------------------------
# Temporal bar chart
# ----------------------------
TEMPORAL_BIN = "year"    # "year" or "decade"

# ----------------------------
# Map extent
# ----------------------------
MAP_EXTENT = (-12.0, 42.5, 30.0, 46.5)  # lon_min, lon_max, lat_min, lat_max

# Downsampling for map only
MAP_MAX_POINTS = {
    "SDN": 70000,
    "ARGO": 90000,
    "ADRIATIC": 25000,
}

RANDOM_SEED = 42

# ----------------------------
# Styles
# ----------------------------
STYLES = {
    "SDN": {
        "marker": "o",
        "s": 6,              # was 5 → slightly larger
        "facecolor": "k",
        "edgecolor": "none",
        "alpha": 0.35,       # was 0.28 → slightly stronger
        "zorder": 3,
        "label": "SeaDataNet",
    },
    "ARGO": {
        "marker": "o",
        "s": 4,              # was 5
        "facecolor": "#8EC5E8",
        "edgecolor": "none",
        "alpha": 0.30,       # was 0.42 → KEY CHANGE
        "zorder": 2,         # lower than SDN
        "label": "Argo",
    },
    "ADRIATIC": {
        "marker": "^",
        "s": 20,
        "facecolor": "#2E8B57",
        "edgecolor": "k",
        "linewidth": 0.25,
        "alpha": 0.80,
        "zorder": 5,
        "label": "Adriatic in-situ",
    },
}

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 16,
})


AX_FACE = "#f0f0f0"
FIG_FACE = "white"
LAND_COLOR = "#d9d6cf"
OCEAN_COLOR = "#f7f8fa"

if HAVE_CMOCEAN:
    HOV_CMAP = cmocean.cm.balance
else:
    HOV_CMAP = mpl.cm.RdBu_r


# ==========================================================
# MAP SPECS
# ==========================================================

@dataclass
class PointSpec:
    name: str
    path: str
    lon_col: str
    lat_col: str
    source_group: str


POINT_SPECS = [
    PointSpec(
        name="SDN",
        path=SDN_MAP_CSV,
        lon_col="Longitude [degrees_east]",
        lat_col="Latitude [degrees_north]",
        source_group="SDN",
    ),
    PointSpec(
        name="ARGO",
        path=ARGO_MAP_CSV,
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        source_group="ARGO",
    ),
]

for fp in ADRIATIC_MAP_CSVS:
    POINT_SPECS.append(
        PointSpec(
            name=Path(fp).stem,
            path=fp,
            lon_col="LONGITUDE",
            lat_col="LATITUDE",
            source_group="ADRIATIC",
        )
    )


# ==========================================================
# HELPERS
# ==========================================================

def maybe_downsample(df: pd.DataFrame, nmax: Optional[int], seed: int) -> pd.DataFrame:
    if nmax is None or len(df) <= nmax:
        return df
    return df.sample(n=nmax, random_state=seed).copy()


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(m):
        return np.nan
    return float(np.sum(values[m] * weights[m]) / np.sum(weights[m]))


def centers_to_edges(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)
    dx = np.diff(x)
    mids = x[:-1] + dx / 2
    left = x[0] - dx[0] / 2
    right = x[-1] + dx[-1] / 2
    return np.concatenate([[left], mids, [right]])


def nice_depth_ticks(zmax: float) -> list[float]:
    base = [0, 100, 200, 300, 500, 700, 1000, 1500, 2000]
    return [z for z in base if z <= zmax]

def get_var_limit(basin: str, var: str) -> float:
    basin = str(basin).upper()
    if var == "T":
        if basin == "NWM":
            return 0.7
        if basin == "MED":
            return 1.0
        return 0.8
    if var == "S":
        return 0.20
    return 0.25


# ==========================================================
# PANEL A: MAP POINTS
# ==========================================================

def load_one_points_dataset(spec: PointSpec) -> pd.DataFrame:
    if not os.path.exists(spec.path):
        raise FileNotFoundError(spec.path)

    df = pd.read_csv(spec.path, usecols=[spec.lon_col, spec.lat_col], low_memory=False)

    df[spec.lon_col] = pd.to_numeric(df[spec.lon_col], errors="coerce")
    df[spec.lat_col] = pd.to_numeric(df[spec.lat_col], errors="coerce")

    out = pd.DataFrame({
        "lon": df[spec.lon_col],
        "lat": df[spec.lat_col],
        "source_group": spec.source_group,
    })

    out = out[np.isfinite(out["lon"]) & np.isfinite(out["lat"])].copy()
    return out


def load_all_points() -> pd.DataFrame:
    parts = []
    for spec in POINT_SPECS:
        print(f"[INFO] Reading map points: {spec.path}")
        parts.append(load_one_points_dataset(spec))
    return pd.concat(parts, ignore_index=True)


# ==========================================================
# PANEL B: UNIQUE CASTS / YEAR (SOURCE-SPECIFIC, EXPLICIT)
# ==========================================================

from collections import defaultdict

# ---- Files for temporal counts ----
# Use the files whose schema is appropriate for cast counting.
SDN_COUNT_FILE = "/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only.csv"
ARGO_COUNT_FILE = "/Users/elena/Projects/IRB/MedClim/Data/Argo/CSV/Argo_with_basins_mainland_special_excluded.csv"

ADRIATIC_COUNT_FILES = [
    "/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/CTD_HR_2019_2023_MFSD_long_fixed_mainland_special_excluded.csv",
    "/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/RB_Northern_Adriatic_Transect_1979_2018_long_fixed_mainland_special_excluded.csv",
    "/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/Palagruza_Transect_1952_2018_long_fixed_mainland_special_excluded.csv",
    "/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/IOR_Adriatic_database_1987_2019_long_fixed_mainland_special_excluded.csv",
]

COUNT_CHUNKSIZE = 400_000


def _clean_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _valid_nonempty(s: pd.Series) -> pd.Series:
    s = _clean_str(s)
    return s.ne("") & s.ne("nan") & s.ne("None")


# -------------------------
# SeaDataNet
# -------------------------
def sdn_build_cast_id(df: pd.DataFrame) -> pd.Series:
    """
    EXACT logic matching the trusted standalone SDN script:
    prefer LOCAL_CDI_ID; fallback to Cruise|Station|date
    """
    cid = df.get("LOCAL_CDI_ID")
    if cid is None and "LOCAL_CDI_ID:METAVAR:INDEXED_TEXT" in df.columns:
        cid = df["LOCAL_CDI_ID:METAVAR:INDEXED_TEXT"]
    if cid is None:
        cid = pd.Series([""] * len(df), index=df.index)

    cid = _clean_str(cid)
    has = cid.ne("") & cid.ne("nan") & cid.ne("None")

    t = pd.to_datetime(df["yyyy-mm-ddThh:mm:ss.sss"], errors="coerce")
    date = t.dt.strftime("%Y-%m-%d").fillna("")

    cruise = _clean_str(df.get("Cruise", pd.Series([""] * len(df), index=df.index)))
    station = _clean_str(df.get("Station:METAVAR:INDEXED_TEXT", pd.Series([""] * len(df), index=df.index)))

    fb = cruise + "|" + station + "|" + date
    return cid.where(has, fb)


def count_sdn_profiles_per_year() -> pd.DataFrame:
    file = SDN_COUNT_FILE
    usecols = [
        "yyyy-mm-ddThh:mm:ss.sss",
        "Depth [m]",
        "LOCAL_CDI_ID",
        "Cruise",
        "Station:METAVAR:INDEXED_TEXT",
    ]

    # allow alt LOCAL_CDI_ID column if present
    header = pd.read_csv(file, nrows=0).columns.tolist()
    if "LOCAL_CDI_ID" not in header and "LOCAL_CDI_ID:METAVAR:INDEXED_TEXT" in header:
        usecols[2] = "LOCAL_CDI_ID:METAVAR:INDEXED_TEXT"

    casts_by_year = defaultdict(set)

    for ch in pd.read_csv(file, usecols=usecols, chunksize=COUNT_CHUNKSIZE, low_memory=False, dtype=str):
        ch = ch.dropna(subset=["yyyy-mm-ddThh:mm:ss.sss", "Depth [m]"])
        if ch.empty:
            continue

        ch["Depth [m]"] = pd.to_numeric(ch["Depth [m]"], errors="coerce")
        ch = ch[ch["Depth [m]"].notna()].copy()
        if ch.empty:
            continue

        dt = pd.to_datetime(ch["yyyy-mm-ddThh:mm:ss.sss"], errors="coerce")
        ch = ch.loc[dt.notna()].copy()
        if ch.empty:
            continue

        ch["year"] = dt.loc[ch.index].dt.year.astype(int)
        ch = ch[(ch["year"] >= YEAR_MIN) & (ch["year"] <= YEAR_MAX)].copy()
        if ch.empty:
            continue

        ch["_cast"] = sdn_build_cast_id(ch)

        yc = ch[["year", "_cast"]].drop_duplicates()
        for y, c in yc.itertuples(index=False):
            casts_by_year[int(y)].add(str(c))

    rows = [{"year": y, "source_group": "SDN", "n": len(casts_by_year[y])} for y in sorted(casts_by_year)]
    return pd.DataFrame(rows)


# -------------------------
# Argo
# -------------------------
def argo_build_cast_id(df: pd.DataFrame) -> pd.Series:
    plat = _clean_str(df.get("PLATFORM_NUMBER", pd.Series([""] * len(df), index=df.index)))
    cyc = _clean_str(df.get("CYCLE_NUMBER", pd.Series([""] * len(df), index=df.index)))
    return plat + "|" + cyc


def count_argo_profiles_per_year() -> pd.DataFrame:
    file = ARGO_COUNT_FILE
    usecols = ["TIME", "PRES", "PLATFORM_NUMBER", "CYCLE_NUMBER", "PRES_QC"]

    header = pd.read_csv(file, nrows=0).columns.tolist()
    usecols = [c for c in usecols if c in header]

    casts_by_year = defaultdict(set)

    for ch in pd.read_csv(file, usecols=usecols, chunksize=COUNT_CHUNKSIZE, low_memory=False, dtype=str):
        if "TIME" not in ch.columns or "PRES" not in ch.columns:
            continue

        ch = ch.dropna(subset=["TIME", "PRES"])
        if ch.empty:
            continue

        ch["PRES"] = pd.to_numeric(ch["PRES"], errors="coerce")
        ch = ch[ch["PRES"].notna()].copy()
        if ch.empty:
            continue

        # optional QC
        if "PRES_QC" in ch.columns:
            q = ch["PRES_QC"].astype(str).str.strip()
            ch = ch[q.isin(["1", "2"])].copy()
            if ch.empty:
                continue

        dt = pd.to_datetime(ch["TIME"], errors="coerce")
        ch = ch.loc[dt.notna()].copy()
        if ch.empty:
            continue

        ch["year"] = dt.loc[ch.index].dt.year.astype(int)
        ch = ch[(ch["year"] >= YEAR_MIN) & (ch["year"] <= YEAR_MAX)].copy()
        if ch.empty:
            continue

        ch["_cast"] = argo_build_cast_id(ch)

        yc = ch[["year", "_cast"]].drop_duplicates()
        for y, c in yc.itertuples(index=False):
            casts_by_year[int(y)].add(str(c))

    rows = [{"year": y, "source_group": "ARGO", "n": len(casts_by_year[y])} for y in sorted(casts_by_year)]
    return pd.DataFrame(rows)


# -------------------------
# Adriatic local datasets
# -------------------------
def adriatic_build_cast_id(df: pd.DataFrame, file_label: str) -> pd.Series:
    station = _clean_str(df.get("STATION", pd.Series([""] * len(df), index=df.index)))
    time = pd.to_datetime(df["TIME"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")

    if "CAST_IDX" in df.columns:
        cast_idx = _clean_str(df["CAST_IDX"])
        return file_label + "|" + station + "|" + cast_idx + "|" + time

    return file_label + "|" + station + "|" + time


def count_one_adriatic_file(file: str) -> pd.DataFrame:
    header = pd.read_csv(file, nrows=0).columns.tolist()

    usecols = ["TIME", "DEPTH_M", "STATION"]
    if "CAST_IDX" in header:
        usecols.append("CAST_IDX")

    casts_by_year = defaultdict(set)
    label = Path(file).stem

    for ch in pd.read_csv(file, usecols=usecols, chunksize=COUNT_CHUNKSIZE, low_memory=False, dtype=str):
        if "TIME" not in ch.columns or "DEPTH_M" not in ch.columns:
            continue

        ch = ch.dropna(subset=["TIME", "DEPTH_M"])
        if ch.empty:
            continue

        ch["DEPTH_M"] = pd.to_numeric(ch["DEPTH_M"], errors="coerce")
        ch = ch[ch["DEPTH_M"].notna()].copy()
        if ch.empty:
            continue

        dt = pd.to_datetime(ch["TIME"], errors="coerce")
        ch = ch.loc[dt.notna()].copy()
        if ch.empty:
            continue

        ch["year"] = dt.loc[ch.index].dt.year.astype(int)
        ch = ch[(ch["year"] >= YEAR_MIN) & (ch["year"] <= YEAR_MAX)].copy()
        if ch.empty:
            continue

        ch["_cast"] = adriatic_build_cast_id(ch, label)

        yc = ch[["year", "_cast"]].drop_duplicates()
        for y, c in yc.itertuples(index=False):
            casts_by_year[int(y)].add(str(c))

    rows = [{"year": y, "source_group": "ADRIATIC", "n": len(casts_by_year[y])} for y in sorted(casts_by_year)]
    return pd.DataFrame(rows)


def count_adriatic_profiles_per_year() -> pd.DataFrame:
    parts = []
    for f in ADRIATIC_COUNT_FILES:
        d = count_one_adriatic_file(f)
        if not d.empty:
            parts.append(d)

    if not parts:
        return pd.DataFrame(columns=["year", "source_group", "n"])

    d = pd.concat(parts, ignore_index=True)
    d = d.groupby(["year", "source_group"], as_index=False)["n"].sum()
    return d


def load_temporal_counts_from_sources() -> pd.DataFrame:
    parts = [
        count_sdn_profiles_per_year(),
        count_argo_profiles_per_year(),
        count_adriatic_profiles_per_year(),
    ]
    parts = [p for p in parts if not p.empty]

    if not parts:
        return pd.DataFrame(columns=["time_bin", "source_group", "n"])

    d = pd.concat(parts, ignore_index=True)

    if TEMPORAL_BIN == "decade":
        d["time_bin"] = (d["year"] // 10) * 10
        d = d.groupby(["time_bin", "source_group"], as_index=False)["n"].sum()
    else:
        d["time_bin"] = d["year"]

    return d


def draw_temporal_panel(ax, d_counts: pd.DataFrame) -> None:
    ax.set_title("b  Temporal evolution of observations", loc="left")
    ax.set_ylabel("Number of unique profiles", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

    if d_counts.empty:
        ax.text(0.5, 0.5, "No valid annual counts produced",
                ha="center", va="center", transform=ax.transAxes)
        return

    if TEMPORAL_BIN == "year":
        x_all = np.arange(YEAR_MIN, YEAR_MAX + 1)
        width = 0.85
        ax.set_xlim(YEAR_MIN - 1, YEAR_MAX + 1)
        ax.set_xticks(np.arange(1950, 2030, 10))
    else:
        x_all = np.arange((YEAR_MIN // 10) * 10, ((YEAR_MAX // 10) * 10) + 10, 10)
        width = 8.0
        ax.set_xlim(x_all.min() - 2, x_all.max() + 2)
        ax.set_xticks(x_all)

    base = np.zeros_like(x_all, dtype=float)

    order = [
        ("SDN", "0.35"),
        ("ADRIATIC", "#4A8F6B"),
        ("ARGO", "#8EC5E8"),
    ]

    for grp, color in order:
        s = d_counts[d_counts["source_group"] == grp].set_index("time_bin")["n"]
        y = np.array([float(s.get(x, 0.0)) for x in x_all], dtype=float)

        ax.bar(
            x_all,
            y,
            bottom=base,
            width=width,
            color=color,
            edgecolor="none",
            label=STYLES[grp]["label"]
        )
        base += y

    ax.legend(loc="upper left", frameon=False)

# ==========================================================
# PANELS C-D: HOVMÖLLERS
# ==========================================================

def load_monthly_composed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])

    req = ["basin", "z_m", "var", "time", "year", "month", "value", "n"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    df["basin"] = df["basin"].astype(str).str.upper().str.strip()
    df["var"] = df["var"].astype(str).str.upper().str.strip()
    df["z_m"] = pd.to_numeric(df["z_m"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")

    df = df[
        df["basin"].notna() &
        df["var"].isin(["T", "S", "SIG0"]) &
        np.isfinite(df["z_m"]) &
        np.isfinite(df["year"]) &
        np.isfinite(df["month"])
    ].copy()

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    df = df[df["z_m"] <= MAX_DEPTH_M].copy()

    return df


def monthly_to_yearly(dfm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (basin, z_m, var, year), g in dfm.groupby(["basin", "z_m", "var", "year"], sort=False):
        gm = g[np.isfinite(g["value"]) & np.isfinite(g["n"]) & (g["n"] > 0)].copy()
        if gm.empty:
            continue

        n_months = gm["month"].nunique()
        if n_months < MIN_VALID_MONTHS_PER_YEAR:
            continue

        val = weighted_mean(gm["value"].to_numpy(), gm["n"].to_numpy())
        nsum = float(gm["n"].sum())

        if not np.isfinite(val):
            continue
        if not np.isfinite(nsum) or nsum < MIN_N_PER_YEAR:
            continue

        rows.append({
            "basin": basin,
            "z_m": float(z_m),
            "var": var,
            "year": int(year),
            "value": float(val),
            "n": float(nsum),
            "n_months": int(n_months),
        })

    return pd.DataFrame(rows)


def add_overall_med(dfy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (z_m, var, year), g in dfy.groupby(["z_m", "var", "year"], sort=False):
        v = g["value"].to_numpy(float)
        w = g["n"].to_numpy(float)
        val = weighted_mean(v, w)
        nsum = float(np.nansum(w[np.isfinite(w) & (w > 0)]))
        if not np.isfinite(val) or nsum <= 0:
            continue
        rows.append({
            "basin": "MED",
            "z_m": float(z_m),
            "var": str(var).upper(),
            "year": int(year),
            "value": float(val),
            "n": float(nsum),
            "n_months": np.nan,
        })
    return pd.DataFrame(rows)


def compute_reference(dfy: pd.DataFrame) -> pd.DataFrame:
    dref = dfy[(dfy["year"] >= REF_YEAR_MIN) & (dfy["year"] <= REF_YEAR_MAX)].copy()

    rows = []
    for (basin, z_m, var), g in dref.groupby(["basin", "z_m", "var"], sort=False):
        ref = weighted_mean(g["value"].to_numpy(), g["n"].to_numpy())
        nref = float(np.nansum(g["n"].to_numpy()))
        if not np.isfinite(ref):
            continue
        rows.append({
            "basin": basin,
            "z_m": float(z_m),
            "var": var,
            "ref_value": float(ref),
            "ref_n": float(nref),
        })
    return pd.DataFrame(rows)


def compute_anomalies(dfy: pd.DataFrame, dref: pd.DataFrame) -> pd.DataFrame:
    da = dfy.merge(dref, on=["basin", "z_m", "var"], how="left")
    da["anom"] = da["value"] - da["ref_value"]
    da = da[np.isfinite(da["anom"])].copy()
    return da


def apply_rolling(da: pd.DataFrame) -> pd.DataFrame:
    if not APPLY_ROLLING or ROLLING_YEARS <= 1:
        return da.copy()

    out = []
    for (basin, z_m, var), g in da.groupby(["basin", "z_m", "var"], sort=False):
        g = g.sort_values("year").copy()
        g["anom"] = (
            g["anom"]
            .rolling(window=ROLLING_YEARS, center=ROLLING_CENTER, min_periods=1)
            .mean()
        )
        out.append(g)
    return pd.concat(out, ignore_index=True)


def basin_specific_zmax(da: pd.DataFrame, basin: str) -> float:
    g = da[da["basin"] == basin].copy()
    if g.empty:
        return MAX_DEPTH_M

    valid_depths = []
    for z_m, gz in g.groupby("z_m"):
        ok = False
        for _, gv in gz.groupby("var"):
            n_valid = np.isfinite(gv["anom"]).sum()
            if n_valid >= MIN_VALID_YEARS_FOR_ZMAX:
                ok = True
                break
        if ok:
            valid_depths.append(float(z_m))

    if not valid_depths:
        return MAX_DEPTH_M

    return min(MAX_DEPTH_M, max(valid_depths))


def pivot_for_plot(da: pd.DataFrame, basin: str, var: str, zmax: Optional[float] = None) -> pd.DataFrame:
    g = da[(da["basin"] == basin) & (da["var"] == var)].copy()
    if zmax is not None:
        g = g[g["z_m"] <= zmax].copy()

    if g.empty:
        return pd.DataFrame()

    piv = g.pivot_table(index="z_m", columns="year", values="anom", aggfunc="mean")
    piv = piv.sort_index().sort_index(axis=1)
    return piv


# ==========================================================
# PLOTTING
# ==========================================================

def draw_map_panel(ax, df_points: pd.DataFrame) -> None:
    lon_min, lon_max, lat_min, lat_max = MAP_EXTENT

    if HAVE_CARTOPY:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=OCEAN_COLOR, zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND_COLOR, edgecolor="0.5", linewidth=0.3, zorder=1)
        ax.coastlines(resolution="50m", linewidth=0.4, color="0.35", zorder=2)
        try:
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.2, edgecolor="0.5", zorder=2)
        except Exception:
            pass
        ax.gridlines(draw_labels=False, linewidth=0.3, color="0.7", alpha=0.35, linestyle="--")
    else:
        ax.set_facecolor("#eef1f4")
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.grid(True, alpha=0.18, linewidth=0.4)

    rng = np.random.default_rng(RANDOM_SEED)

    for grp in ["ARGO", "SDN", "ADRIATIC"]:
        g = df_points[df_points["source_group"] == grp].copy()
        if g.empty:
            continue
        g = maybe_downsample(g, MAP_MAX_POINTS.get(grp), seed=int(rng.integers(0, 1_000_000)))
        st = STYLES[grp].copy()
        label = st.pop("label")
        if HAVE_CARTOPY:
            ax.scatter(g["lon"], g["lat"], transform=ccrs.PlateCarree(), label=label, **st)
        else:
            ax.scatter(g["lon"], g["lat"], label=label, **st)

    ax.set_title("a  Spatial distribution of observations", loc="left", pad=6, fontsize=14, fontweight="bold")
    if HAVE_CARTOPY:
        ax.set_xticks([-10, 0, 10, 20, 30, 40], crs=ccrs.PlateCarree())
        ax.set_yticks([30, 35, 40, 45], crs=ccrs.PlateCarree())
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o", linestyle="None",
            markerfacecolor="k", markeredgecolor="none",
            markersize=8, label="SeaDataNet"
        ),
        Line2D(
            [0], [0],
            marker="^", linestyle="None",
            markerfacecolor="#2E8B57", markeredgecolor="k",
            markeredgewidth=0.4, markersize=9, label="Adriatic in-situ"
        ),
        Line2D(
            [0], [0],
            marker="o", linestyle="None",
            markerfacecolor="#8EC5E8", markeredgecolor="none",
            markersize=8, label="Argo"
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        fontsize=13,
        borderpad=0.8,
        labelspacing=0.5,
        handlelength=1.2
    )

def draw_temporal_panel(ax, d_counts: pd.DataFrame) -> None:
    ax.set_title("b  Temporal evolution of observations", loc="left", pad=6, fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of unique profiles", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

    if d_counts.empty:
        ax.text(0.5, 0.5, "No valid annual counts produced", ha="center", va="center", transform=ax.transAxes)
        return

    if TEMPORAL_BIN == "year":
        x_all = np.arange(YEAR_MIN, YEAR_MAX + 1)
        width = 0.85
        ax.set_xlim(YEAR_MIN - 1, YEAR_MAX + 1)
        ax.set_xticks(np.arange(1950, 2030, 10))
    else:
        x_all = np.arange((YEAR_MIN // 10) * 10, ((YEAR_MAX // 10) * 10) + 10, 10)
        width = 8.0
        ax.set_xlim(x_all.min() - 2, x_all.max() + 2)
        ax.set_xticks(x_all)

    base = np.zeros_like(x_all, dtype=float)

    order = [
        ("SDN", "0.35"),
        ("ADRIATIC", "#4A8F6B"),
        ("ARGO", "#8EC5E8"),
    ]

    for grp, color in order:
        s = d_counts[d_counts["source_group"] == grp].set_index("time_bin")["n"]
        y = np.array([float(s.get(x, 0.0)) for x in x_all], dtype=float)
        ax.bar(
            x_all, y, bottom=base, width=width,
            color=color, edgecolor="none", label=STYLES[grp]["label"]
        )
        base += y

    ax.tick_params(axis="both", which="major", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2, width=0.6)

    ax.legend(loc="upper left", frameon=False)


def draw_hov_panel(ax, da: pd.DataFrame, basin: str, panel_label: str, subtitle: str,
                    show_ylabel: bool = True, show_cbar_label: bool = True) -> None:
    zmax = basin_specific_zmax(da, basin)
    piv = pivot_for_plot(da, basin=basin, var=HOV_VAR, zmax=zmax)

    if piv.empty:
        ax.text(0.5, 0.5, f"No data for {basin}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    Z = piv.to_numpy(dtype=float)
    depths = piv.index.to_numpy(dtype=float)
    years = piv.columns.to_numpy(dtype=float)

    x_edges = centers_to_edges(years)
    y_edges = centers_to_edges(depths)

    lim = get_var_limit(basin, HOV_VAR)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    cmap = mpl.cm.get_cmap(HOV_CMAP).copy() if hasattr(mpl.cm, "get_cmap") else HOV_CMAP
    try:
        cmap.set_bad(AX_FACE)
    except Exception:
        pass

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(Z),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    ax.set_facecolor(AX_FACE)
    ax.invert_yaxis()
    ax.set_title(f"{panel_label}  {subtitle}", loc="left")
    ax.set_xlabel("Year", fontsize=12)
    if show_ylabel:
        ax.set_ylabel("Depth (m)", fontsize=12)
    else:
        ax.set_ylabel("")
    ax.set_xlim(YEAR_MIN - 0.5, YEAR_MAX + 0.5)
    ax.set_ylim(zmax, 0)

    yt = nice_depth_ticks(zmax)
    if len(yt) >= 2:
        ax.set_yticks(yt)

    ax.tick_params(axis="both", which="major", labelsize=11)

    cbar = plt.colorbar(pcm, ax=ax, pad=0.02, fraction=0.045, aspect=24)
    if show_cbar_label:
        cbar.set_label("Temperature anomaly (°C)", fontsize=12)
        cbar.ax.tick_params(labelsize=12)
    else:
        cbar.set_label("")
    cbar.ax.tick_params(labelsize=10)


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)

    print("[INFO] Loading map points")
    df_points = load_all_points()

    print("[INFO] Loading annual unique-profile counts")
    d_counts = load_temporal_counts_from_sources()

    print("[INFO] Loading monthly composed basin series")
    dfm = load_monthly_composed(MONTHLY_COMPOSED_CSV)

    print("[INFO] Aggregating monthly -> yearly")
    dfy = monthly_to_yearly(dfm)
    if dfy.empty:
        raise RuntimeError("No yearly data produced from monthly_timeseries_all_bins_composed.csv")

    print("[INFO] Building overall MED")
    dmed = add_overall_med(dfy)
    if not dmed.empty:
        dfy = pd.concat([dfy, dmed], ignore_index=True)

    print("[INFO] Computing reference")
    dref = compute_reference(dfy)
    if dref.empty:
        raise RuntimeError("Reference climatology is empty.")

    print("[INFO] Computing anomalies")
    da = compute_anomalies(dfy, dref)
    if da.empty:
        raise RuntimeError("No anomalies produced.")

    if APPLY_ROLLING:
        print(f"[INFO] Applying {ROLLING_YEARS}-year rolling mean")
        da = apply_rolling(da)

    print("[INFO] Creating figure")

    fig = plt.figure(figsize=(14, 12), facecolor=FIG_FACE)
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.55, 0.92, 1.12],
        hspace=0.28,
        wspace=0.16
    )

    if HAVE_CARTOPY:
        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    else:
        ax_map = fig.add_subplot(gs[0, :])

    ax_bar = fig.add_subplot(gs[1, :])
    ax_med = fig.add_subplot(gs[2, 0])
    ax_nwm = fig.add_subplot(gs[2, 1])

    fig.subplots_adjust(left=0.055, right=0.985, top=0.93, bottom=0.06)

    draw_map_panel(ax_map, df_points)
    d_counts = load_temporal_counts_from_sources()
    draw_temporal_panel(ax_bar, d_counts)
    draw_hov_panel(
        ax_med, da,
        basin="MED",
        panel_label="c",
        subtitle="Whole Mediterranean temperature anomaly",
        show_ylabel=True,
        show_cbar_label=False,
    )

    draw_hov_panel(
        ax_nwm, da,
        basin=REPRESENTATIVE_BASIN,
        panel_label="d",
        subtitle=f"{REPRESENTATIVE_BASIN} temperature anomaly",
        show_ylabel=False,
        show_cbar_label=True,
    )

    #fig.suptitle(
    #    "Figure 1. Observational data set and coverage",
    #    y=0.975,
    #    fontsize=17,
    #    fontweight="bold"
    #)
    footer = (
        "Panel (b): annual counts are unique profiles/casts per source. "
        "Panels (c-d): yearly temperature anomalies relative to 1950-1999"
    )
    if APPLY_ROLLING:
        footer += f"; {ROLLING_YEARS}-year smoothing applied."

    # Comment the footer - use the text as a caption instead
    #fig.text(0.5, 0.015, footer, ha="center", va="bottom", fontsize=12)

    fig.savefig(OUTPNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPDF, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote {OUTPNG}")
    print(f"[OK] Wrote {OUTPDF}")


if __name__ == "__main__":
    main()