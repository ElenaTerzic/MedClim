#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reads : Stats/MultiSource_Climatology_BINNED_ELIG/*/POOLED_RAW/climatology_POOLED_RAW_*_after_sigma.csv, Stats/cast_counts_bybin_by_platform_season_period_T.csv, Stats/points_metadata_all_sources_ALL_*.csv
# Writes: Plots/MultiPlatform/AllBasins_Publication/<mode>/<basin>_<mode>_publication_3row_PDA_maps_support.png
"""
plot_manca_all_basins_publication_SIG0_maps_support.py

Generalizes Elena's DJ1/DJ3 publication-ready figure to ALL Manca basins
(atomic and/or composed), keeping the exact same layout and logic.

Layout:
- 3 rows: 21c, 20c, Δ (21c−20c)
- 4 columns: Temperature, Salinity, PDA (σ0), Maps/Support
- PDA uses SIG0_mean/SIG0_std from climatology CSVs
- Maps: row1 col4 = 21c, row2 col4 = 20c
- Support panel: row3 col4

Map extent priority:
1) EXTENT_OVERRIDE (manual)
2) Basins.manca polygon/rectangle bounds (+ padding)
3) points metadata bounds (+ padding)
4) EXTENT_DEFAULT fallback

Run examples:
  python plot_manca_all_basins.py --mode atomic
  python plot_manca_all_basins.py --mode composed
  python plot_manca_all_basins.py --mode both
  python plot_manca_all_basins.py --mode atomic --basins DJ1 DJ3 DF1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Iterable, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Scripts.paths import STATS, PLOTS

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False


# =========================
# USER CONFIG (EDIT ME)
# =========================

# --- CLIMATOLOGY CSVs (atomic) ---
CLIM_21C_ATOMIC_CSV = str(STATS / "MultiSource_Climatology_BINNED_ELIG" / "21c_2000_2025" / "POOLED_RAW" / "climatology_POOLED_RAW_21c_2000_2025_atomic_after_sigma.csv")
CLIM_20C_ATOMIC_CSV = str(STATS / "MultiSource_Climatology_BINNED_ELIG" / "20c_1950_1999" / "POOLED_RAW" / "climatology_POOLED_RAW_20c_1950_1999_atomic_after_sigma.csv")

# --- CLIMATOLOGY CSVs (composed) ---
CLIM_21C_COMPOSED_CSV = str(STATS / "MultiSource_Climatology_BINNED_ELIG" / "21c_2000_2025" / "POOLED_RAW" / "climatology_POOLED_RAW_21c_2000_2025_composed_after_sigma.csv")
CLIM_20C_COMPOSED_CSV = str(STATS / "MultiSource_Climatology_BINNED_ELIG" / "20c_1950_1999" / "POOLED_RAW" / "climatology_POOLED_RAW_20c_1950_1999_composed_after_sigma.csv")

# --- SUPPORT COUNTS ---
COUNTS_CSV = str(STATS / "cast_counts_bybin_by_platform_season_period_T.csv")

# --- POINTS METADATA (for maps) ---
POINTS_META_ATOMIC_CSV = str(STATS / "points_metadata_all_sources_ALL_atomic.csv")
POINTS_META_COMPOSED_CSV = str(STATS / "points_metadata_all_sources_ALL_composed.csv")

# --- OUTPUT ---
OUTDIR = str(PLOTS / "MultiPlatform" / "AllBasins_Publication")

# Plotting-only sanity masks / overrides
SALINITY_PLOT_RANGE = (30.0, 40.5)  # [psu]

ZMAX_OVERRIDE = {
    "DJ1":  100.0,  # plotting cap per basin (optional)
    "DJ2":  250.0,  # plotting cap per basin (optional)
    "MADR": 250.0,  # Middle Adriatic (composed)
    "NADR": 80.0,  # North Adriatic (composed)
}

FULL_DEPTH_BASINS = {"DJ3"}  # force full-depth axis even if deep bins are NaN

DISABLE_LOWCOV_MASK_BASINS = {"DJ3"}

GAP_BREAK_BASINS = {"DJ1"}
GAP_M = 40.0

# Optional manual map extents (lon_min, lon_max, lat_min, lat_max)
EXTENT_OVERRIDE = {
    # "DJ3": (13.8, 20.6, 38.0, 44.0),
}

# Default Med-wide fallback extent
EXTENT_DEFAULT = (-6.5, 37.0, 30.0, 46.5)

# Padding in degrees added around basin bbox
EXTENT_PAD_DEG = 0.5

# seasons/colors (match Med)
SEASON_ORDER = ["JFM", "AMJ", "JAS", "OND"]
SEASON_LABEL = {"JFM": "Winter", "AMJ": "Spring", "JAS": "Summer", "OND": "Autumn"}
SEASON_COLOR = {"JFM": "C0", "AMJ": "C2", "JAS": "C3", "OND": "C1"}

# Insets (0–200 m) for deep basins (same as your DJ script)
# ==========================================================
# Inset defaults
# ==========================================================

BR = (0.55, 0.12, 0.40, 0.40)
BL = (0.12, 0.12, 0.40, 0.40)

INSET_DEFAULT_BBOX = {
    ("T",    "21c"): BR,
    ("T",    "20c"): BR,
    ("S",    "21c"): BL,
    ("S",    "20c"): BL,
    ("SIG0", "21c"): BL,
    ("SIG0", "20c"): BL,
}

# ADD THIS:
INSET_DEFAULT_BBOX.update({
    ("dT",    "dlt"): BR,
    ("dS",    "dlt"): BR,
    ("dSIG0", "dlt"): BR,
})

INSET_BBOX = {}

for B in ["DH2", "DH3", "DL1", "DL4"]:
    INSET_BBOX[(B, "S", "21c")] = BR
    INSET_BBOX[(B, "S", "20c")] = BR

for B in ["DL2", "DL3"]:
    INSET_BBOX[(B, "S", "21c")] = BR
    INSET_BBOX[(B, "S", "20c")] = BR
    INSET_BBOX[(B, "dS", "dlt")] = BL  # delta salinity

for B in ["DJ2", "DT2"]:
    INSET_BBOX[(B, "dT", "dlt")] = BL  # delta salinity

for B in ["LEV", "SAEG"]:
    INSET_BBOX[(B, "S", "21c")] = BR
    INSET_BBOX[(B, "S", "20c")] = BR

# WEST delta salinity (3rd row) -> bottom-left
INSET_BBOX[("WEST", "dS", "dlt")] = BL

NO_INSET_BASINS = {"MADR", "NADR"}


def get_inset_bbox(basin: str, var_key: str, row_key: str):
    basin = str(basin).upper()
    return INSET_BBOX.get((basin, var_key, row_key), INSET_DEFAULT_BBOX[(var_key, row_key)])

DELTA_SIG0_XLIM = (-1.0, 1.0)

PLATFORM_NORMALIZE = {
    "SDN": "SDN",
    "ARGO": "ARGO",
    # everything else collapses to ADRIATIC group in support plot (as in your DJ script)
}

MIN_N = 10
APPLY_MIN_N_BASINS = {"DJ1"}  # keep same as your DJ script

LINESTYLE_21C = "-"
SUPPORT_LINESTYLE_21C = "-"
SUPPORT_LINESTYLE_20C = "--"
LINESTYLE_20C = "-"  # profiles solid; dashed only in support panel

# =========================
# Helpers
# =========================

def _must_have_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Have: {list(df.columns)[:80]}")

def load_climatology(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalize for common variants
    if "basin" in df.columns and "BASIN" not in df.columns:
        df.rename(columns={"basin": "BASIN"}, inplace=True)

    req = ["BASIN", "season", "z_m", "T_mean", "T_std", "S_mean", "S_std"]
    _must_have_cols(df, req)

    if "sig0_mean" in df.columns and "SIG0_mean" not in df.columns:
        df.rename(columns={"sig0_mean": "SIG0_mean", "sig0_std": "SIG0_std"}, inplace=True)

    _must_have_cols(df, ["SIG0_mean", "SIG0_std"])
    df["season"] = df["season"].astype(str).str.strip()
    df["BASIN"] = df["BASIN"].astype(str).str.strip().str.upper()
    df["z_m"] = pd.to_numeric(df["z_m"], errors="coerce")

    return df

def is_adriatic_basin(basin: str) -> bool:
    """True only for the *Adriatic* basins (DJ1–DJ3) and their composed aliases.

    NOTE: Many Eastern-Med basins are DJ* (DJ4–DJ8) but are **Ionian**, not Adriatic.
    """
    b = str(basin).upper().strip()
    if b in {"DJ1", "DJ2", "DJ3"}:
        return True
    # composed Adriatic (from manca.py)
    if b in {"NADR", "MADR", "SADR"}:
        return True
    return False

def apply_min_n_mask(d: pd.DataFrame, basin: str) -> pd.DataFrame:
    b = str(basin).upper()
    if MIN_N is None or MIN_N <= 1 or b not in APPLY_MIN_N_BASINS:
        return d
    out = d.copy()
    # Temperature
    if "n_T" in out.columns:
        bad = pd.to_numeric(out["n_T"], errors="coerce") < MIN_N
        out.loc[bad.fillna(False), ["T_mean", "T_std"]] = np.nan
        if "n_SIG0" not in out.columns:
            out.loc[bad.fillna(False), ["SIG0_mean", "SIG0_std"]] = np.nan
    # Salinity
    if "n_S" in out.columns:
        bad = pd.to_numeric(out["n_S"], errors="coerce") < MIN_N
        out.loc[bad.fillna(False), ["S_mean", "S_std"]] = np.nan
    # σ0
    if "n_SIG0" in out.columns:
        bad = pd.to_numeric(out["n_SIG0"], errors="coerce") < MIN_N
        out.loc[bad.fillna(False), ["SIG0_mean", "SIG0_std"]] = np.nan
    return out

def get_manca_members(basin: str) -> list[str]:
    """
    For composed basins (WEST/ION/LEV/...), return list of member atomic basin codes (DF*, DI*, DJ*, DL*, DS*, DT*, DH*).
    Returns [] if not available.
    """
    b = str(basin).strip().lower()
    try:
        import Basins  # type: ignore
        if not hasattr(Basins, "manca") or not hasattr(Basins.manca, b):
            return []
        obj = getattr(Basins.manca, b)
    except Exception:
        return []

    for attr in ("members", "basins", "children", "components"):
        if hasattr(obj, attr):
            try:
                mem = getattr(obj, attr)
                out = []
                for m in mem:
                    if hasattr(m, "abbrev"):
                        out.append(str(m.abbrev).upper())
                out = sorted(set(out))
                # keep only atomic D?? basins
                out = [x for x in out if x.startswith("D") and len(x) == 3]
                return out
            except Exception:
                pass

    return []


def subset_counts_for_basin_or_members(dfc: pd.DataFrame, basin: str) -> pd.DataFrame:
    """
    Try direct basin counts first. If empty and basin is composed, sum member basins.
    Keeps all columns (including sdn_class) so summarize_counts() can do its normal job.
    """
    b = str(basin).upper()

    bcol = "basin" if "basin" in dfc.columns else ("BASIN" if "BASIN" in dfc.columns else None)
    if bcol is None:
        return pd.DataFrame()

    dfc[bcol] = dfc[bcol].astype(str).str.strip().str.upper()

    direct = dfc[dfc[bcol] == b].copy()
    if not direct.empty:
        return direct

    members = get_manca_members(b)
    if not members:
        return pd.DataFrame()

    sub = dfc[dfc[bcol].isin(members)].copy()
    if sub.empty:
        return pd.DataFrame()

    # IMPORTANT: for composed, we need to SUM counts across members *within each depth bin*
    group_cols = []
    for c in ("period", "source", "season", "z_m", "sdn_class", "SDN_CLASS"):
        if c in sub.columns:
            # normalize SDN_CLASS naming
            if c == "SDN_CLASS":
                sub.rename(columns={"SDN_CLASS": "sdn_class"}, inplace=True)
                c = "sdn_class"
            group_cols.append(c)

    # choose the count column
    ncol = "n_casts" if "n_casts" in sub.columns else ("n" if "n" in sub.columns else ("count" if "count" in sub.columns else None))
    if ncol is None:
        # fallback to first numeric
        numcols = [c for c in sub.columns if c not in group_cols and pd.api.types.is_numeric_dtype(sub[c])]
        if not numcols:
            return pd.DataFrame()
        ncol = numcols[0]

    out = sub.groupby(group_cols, dropna=False, as_index=False)[ncol].sum()

    # Put the composed basin name back in for downstream labeling (optional)
    out[bcol] = b
    return out

def zmax_valid_for_basin(df21: pd.DataFrame, df20: pd.DataFrame, basin: str) -> float:
    d21 = apply_min_n_mask(df21[df21["BASIN"] == basin].copy(), basin)
    d20 = apply_min_n_mask(df20[df20["BASIN"] == basin].copy(), basin)

    if str(basin).upper() in FULL_DEPTH_BASINS:
        z_all = pd.concat([d21["z_m"], d20["z_m"]], ignore_index=True).dropna()
        zmax = float(z_all.max()) if len(z_all) else 0.0
    else:
        def _valid(d):
            ok = np.isfinite(d["T_mean"]) | np.isfinite(d["S_mean"]) | np.isfinite(d["SIG0_mean"])
            return d.loc[ok, "z_m"]

        z = pd.concat([_valid(d21), _valid(d20)], ignore_index=True).dropna()
        if len(z) == 0:
            zmax = float(pd.concat([d21["z_m"], d20["z_m"]], ignore_index=True).max())
        else:
            zmax = float(z.max())

    zov = ZMAX_OVERRIDE.get(str(basin).upper())
    if zov is not None:
        zmax = float(min(zmax, zov))
    return zmax

def compute_delta(df21: pd.DataFrame, df20: pd.DataFrame, basin: str) -> pd.DataFrame:
    a = apply_min_n_mask(df21[df21["BASIN"] == basin].copy(), basin)
    b = apply_min_n_mask(df20[df20["BASIN"] == basin].copy(), basin)
    m = a.merge(b, on=["season", "z_m"], suffixes=("_21", "_20"), how="inner")
    out = pd.DataFrame({
        "season": m["season"],
        "z_m": m["z_m"],
        "dT": m["T_mean_21"] - m["T_mean_20"],
        "dS": m["S_mean_21"] - m["S_mean_20"],
        "dSIG0": m["SIG0_mean_21"] - m["SIG0_mean_20"],
        "dT_std": np.sqrt(m["T_std_21"]**2 + m["T_std_20"]**2),
        "dS_std": np.sqrt(m["S_std_21"]**2 + m["S_std_20"]**2),
        "dSIG0_std": np.sqrt(m["SIG0_std_21"]**2 + m["SIG0_std_20"]**2),
    })
    return out

def _insert_nan_gaps(y: np.ndarray, *arrs: np.ndarray, gap_m: float = 40.0):
    if y.size < 2:
        return (y, *arrs)
    dy = np.diff(y)
    gap_idx = np.where(dy > gap_m)[0]
    if gap_idx.size == 0:
        return (y, *arrs)
    y2 = y.astype(float).copy()
    arrs2 = [a.astype(float).copy() for a in arrs]
    for gi in gap_idx[::-1]:
        y2 = np.insert(y2, gi + 1, np.nan)
        arrs2 = [np.insert(a, gi + 1, np.nan) for a in arrs2]
    return (y2, *arrs2)

def _unify_xlim(ax_top, ax_mid, pad_frac=0.03):
    x1 = ax_top.get_xlim()
    x2 = ax_mid.get_xlim()
    lo = min(x1[0], x2[0])
    hi = max(x1[1], x2[1])
    pad = (hi - lo) * pad_frac
    ax_top.set_xlim(lo - pad, hi + pad)
    ax_mid.set_xlim(lo - pad, hi + pad)
    return (lo - pad, hi + pad)

def plot_profile(ax, df: pd.DataFrame, var: str, std: str, zmax: float, title: str, xlabel: str,
                 show_title: bool = True, linestyle: str = "-", basin: Optional[str] = None):
    for season in SEASON_ORDER:
        d = df[df["season"] == season].sort_values("z_m")
        d = d[d["z_m"] <= zmax]

        # Low-coverage mask (if present)
        if basin is None or str(basin).upper() not in DISABLE_LOWCOV_MASK_BASINS:
            flag_col = None
            if var == "T_mean" and "flag_T_lowcov" in d.columns:
                flag_col = "flag_T_lowcov"
            elif var == "S_mean" and "flag_S_lowcov" in d.columns:
                flag_col = "flag_S_lowcov"
            elif var == "SIG0_mean" and "flag_SIG0_lowcov" in d.columns:
                flag_col = "flag_SIG0_lowcov"
            if flag_col is not None:
                d = d[d[flag_col].fillna(0).astype(int) == 0]

        # Extra salinity sanity mask (plotting only)
        if var == "S_mean" and "S_mean" in d.columns:
            s0, s1 = SALINITY_PLOT_RANGE
            d = d[(d["S_mean"] >= s0) & (d["S_mean"] <= s1)]

        if d.empty or (var not in d.columns):
            continue

        x0 = d[var].to_numpy(dtype=float)
        y0 = d["z_m"].to_numpy(dtype=float)

        m_line = np.isfinite(x0) & np.isfinite(y0)
        if not m_line.any():
            continue
        xL, yL = x0[m_line], y0[m_line]
        if (basin is None) or (str(basin).upper() in GAP_BREAK_BASINS):
            yL, xL = _insert_nan_gaps(yL, xL, gap_m=GAP_M)

        ax.plot(xL, yL, lw=2.0, color=SEASON_COLOR[season], linestyle=linestyle)

        if std in d.columns:
            s0 = d[std].to_numpy(dtype=float)
            m_fill = m_line & np.isfinite(s0)
            if m_fill.any():
                xF, yF, sF = x0[m_fill], y0[m_fill], s0[m_fill]
                if (basin is None) or (str(basin).upper() in GAP_BREAK_BASINS):
                    yF, xF, sF = _insert_nan_gaps(yF, xF, sF, gap_m=GAP_M)
                ax.fill_betweenx(yF, xF - sF, xF + sF, color=SEASON_COLOR[season], alpha=0.15, linewidth=0)

    ax.set_ylim(zmax, 0)
    ax.grid(True, alpha=0.25)
    if show_title:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)

def plot_delta(ax, df: pd.DataFrame, var: str, std: str, zmax: float, xlabel: str, linestyle: str = "-",
               basin: Optional[str] = None):
    for season in SEASON_ORDER:
        d = df[df["season"] == season].sort_values("z_m")
        d = d[d["z_m"] <= zmax]
        if d.empty or (var not in d.columns):
            continue

        x0 = d[var].to_numpy(dtype=float)
        y0 = d["z_m"].to_numpy(dtype=float)

        m_line = np.isfinite(x0) & np.isfinite(y0)
        if not m_line.any():
            continue
        xL, yL = x0[m_line], y0[m_line]
        if (basin is None) or (str(basin).upper() in GAP_BREAK_BASINS):
            yL, xL = _insert_nan_gaps(yL, xL, gap_m=GAP_M)

        ax.plot(xL, yL, lw=2.0, color=SEASON_COLOR[season], linestyle=linestyle)

        if std in d.columns:
            s0 = d[std].to_numpy(dtype=float)
            m_fill = m_line & np.isfinite(s0)
            if m_fill.any():
                xF, yF, sF = x0[m_fill], y0[m_fill], s0[m_fill]
                if (basin is None) or (str(basin).upper() in GAP_BREAK_BASINS):
                    yF, xF, sF = _insert_nan_gaps(yF, xF, sF, gap_m=GAP_M)
                ax.fill_betweenx(yF, xF - sF, xF + sF, color=SEASON_COLOR[season], alpha=0.15, linewidth=0)

    if var == "dSIG0" and DELTA_SIG0_XLIM is not None:
        ax.set_xlim(*DELTA_SIG0_XLIM)

    ax.axvline(0, color="k", lw=1, alpha=0.75)
    ax.set_ylim(zmax, 0)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(xlabel)

# =========================
# Support panel (unchanged from DJ script)
# =========================

def normalize_counts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "basin" in d.columns and "BASIN" not in d.columns:
        d.rename(columns={"basin": "BASIN"}, inplace=True)
    if "source" in d.columns:
        d["source"] = d["source"].astype(str).str.upper()
    d["PERIOD"] = d["period"].astype(str).str.replace("21c_2000_2025", "21c").str.replace("20c_1950_1999", "20c")
    d["season"] = d["season"].astype(str)
    d["season"] = d["season"].apply(normalize_season_label)
    d["z_m"] = pd.to_numeric(d["z_m"], errors="coerce")
    d["n_casts"] = pd.to_numeric(d["n_casts"], errors="coerce")
    return d

def normalize_season_label(s: str) -> str:
    """Map common season labels to our 4-season codes: JFM, AMJ, JAS, OND."""
    x = str(s).strip().upper()
    # already in target set
    if x in {"JFM","AMJ","JAS","OND"}:
        return x
    # common 3-month climatology codes
    if x in {"DJF","WINTER","WIN","W"}:
        return "JFM"
    if x in {"MAM","SPRING","SPR","SP"}:
        return "AMJ"
    if x in {"JJA","SUMMER","SUM","SU"}:
        return "JAS"
    if x in {"SON","AUTUMN","FALL","AUT","FA"}:
        return "OND"
    # numeric seasons (1..4) sometimes used
    if x in {"1","01"}: return "JFM"
    if x in {"2","02"}: return "AMJ"
    if x in {"3","03"}: return "JAS"
    if x in {"4","04"}: return "OND"
    return x

def summarize_counts(dfc: pd.DataFrame, zmax: float) -> pd.DataFrame:
    d = dfc.copy()

    # normalize column names
    ren = {}
    for c in d.columns:
        cl = c.lower()
        if cl == "basin": ren[c] = "basin"
        elif cl == "period": ren[c] = "period"
        elif cl == "source": ren[c] = "source"
        elif cl == "season": ren[c] = "season"
        elif cl in ("z_m", "depth", "z", "zmid"): ren[c] = "z_m"
        elif cl in ("n_casts", "n", "count", "counts"): ren[c] = "n_casts"
        elif cl in ("sdn_class", "platform", "ptype", "instrument"): ren[c] = "sdn_class"
    d.rename(columns=ren, inplace=True)

    _must_have_cols(d, ["period", "source", "season", "z_m", "n_casts"])

    d["period"] = d["period"].astype(str)
    d["source"] = d["source"].astype(str)
    d["season"] = d["season"].astype(str)
    d["season"] = d["season"].apply(normalize_season_label)
    d["z_m"] = pd.to_numeric(d["z_m"], errors="coerce")
    d["n_casts"] = pd.to_numeric(d["n_casts"], errors="coerce")
    if "sdn_class" in d.columns:
        d["sdn_class"] = d["sdn_class"].astype(str)

    d = d[np.isfinite(d["z_m"]) & np.isfinite(d["n_casts"])].copy()
    d = d[d["z_m"] <= float(zmax)].copy()
    if d.empty:
        return pd.DataFrame(columns=["period", "source", "season", "p25", "p75", "median", "nmax"])

    # SUM across SDN classes within each depth bin (if present)
    gcols = ["period", "source", "season", "z_m"]
    d = d.groupby(gcols, dropna=False, as_index=False)["n_casts"].sum()

    out = []
    for (per, src, seas), g in d.groupby(["period", "source", "season"], dropna=False):
        n = g["n_casts"].to_numpy(float)
        n = n[np.isfinite(n)]
        if n.size == 0:
            continue
        out.append({
            "period": str(per),
            "source": str(src),
            "season": str(seas),
            "p25": float(np.nanpercentile(n, 25)),
            "p75": float(np.nanpercentile(n, 75)),
            "median": float(np.nanmedian(n)),
            "nmax": float(np.nanmax(n)),
        })
    return pd.DataFrame(out)

def plot_support(ax, counts_csv: str, basin: str, zmax: float):
    stat = None
    if counts_csv and Path(counts_csv).exists() and Path(counts_csv).stat().st_size > 0:
        try:
            dfc = pd.read_csv(counts_csv)

            # NEW: direct basin or composed-as-sum(members)
            dfc = subset_counts_for_basin_or_members(dfc, basin)

            if len(dfc):
                stat = summarize_counts(dfc, zmax=float(zmax))
        except Exception:
            stat = None

    if stat is None or stat.empty:
        ax.text(0.5, 0.5, "No counts CSV", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    stat["period_short"] = stat["period"].astype(str)
    stat["period_short"] = stat["period_short"].str.replace("21c_2000_2025", "21c", regex=False)
    stat["period_short"] = stat["period_short"].str.replace("20c_1950_1999", "20c", regex=False)

    def _support_source_group(s: str) -> str:
        s = str(s).upper()
        if "ARGO" in s:
            return "ARGO"
        if "SDN" in s or "SEADATANET" in s:
            return "SDN"
        if "ADRI" in s:
            return "ADRIATIC"
        # everything else: treat as in-situ bucket (kept only for Adriatic basins below)
        return "ADRIATIC"

    stat["source_norm"] = stat["source"].astype(str).apply(_support_source_group)

    # Drop Adriatic contribution for non-Adriatic basins
    if not is_adriatic_basin(basin):
        stat = stat[stat["source_norm"] != "ADRIATIC"].copy()
        if stat.empty:
            ax.text(0.5, 0.5, "No ARGO/SDN support", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return

    if is_adriatic_basin(basin):
        order = ["ARGO", "SDN", "ADRIATIC"]
        colors = {"ARGO": "C1", "SDN": "C0", "ADRIATIC": "C4"}
    else:
        order = ["ARGO", "SDN"]
        colors = {"ARGO": "C1", "SDN": "C0"}

    x = np.arange(len(SEASON_ORDER))
    dx = 0.22
    dp = 0.06

    for ip, src in enumerate(order):
        ss = stat[stat["source_norm"] == src]
        if ss.empty:
            continue
        for i, season in enumerate(SEASON_ORDER):
            sss = ss[ss["season"] == season]
            if sss.empty:
                continue
            xc0 = x[i] + (ip - 1) * dx
            for per, ls, xo in [("21c", SUPPORT_LINESTYLE_21C, +dp), ("20c", SUPPORT_LINESTYLE_20C, -dp)]:
                row = sss[sss["period_short"] == per]
                if row.empty:
                    continue
                row = row.iloc[0]
                xc = xc0 + xo
                ax.vlines(xc, row["p25"], row["p75"], colors=colors[src], linestyles=ls, lw=1.8, alpha=0.9)

                mfc = colors[src] if per == "21c" else "none"
                mec = colors[src]
                ax.plot([xc], [row["median"]], marker="o", ms=5, mfc=mfc, mec=mec, color=mec, alpha=0.9, linestyle="None")
                ax.plot([xc], [row["nmax"]], marker="s", ms=5, mfc=mfc, mec=mec, color=mec, alpha=0.9, linestyle="None")

    ax.set_xticks(x)
    ax.set_xticklabels([SEASON_LABEL[s] for s in SEASON_ORDER], rotation=0)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Casts per depth-bin (n)", labelpad=8)
    ax.grid(True, alpha=0.25)

    import matplotlib.lines as mlines
    h_src = [
        mlines.Line2D([], [], color=colors["ARGO"], lw=2, label="ARGO"),
        mlines.Line2D([], [], color=colors["SDN"],  lw=2, label="SDN"),
    ]
    if is_adriatic_basin(basin):
        h_src.append(mlines.Line2D([], [], color=colors["ADRIATIC"], lw=2, label="ADRIATIC"))
    
    h_per = [
        mlines.Line2D([], [], color="k", lw=2, linestyle=SUPPORT_LINESTYLE_21C, label="21st c."),
        mlines.Line2D([], [], color="k", lw=2, linestyle=SUPPORT_LINESTYLE_20C, label="20th c."),
    ]
    leg1 = ax.legend(handles=h_src, loc="upper right", frameon=False, fontsize=9, handlelength=2.5)
    ax.add_artist(leg1)
    ax.legend(handles=h_per, loc="upper left", frameon=False, fontsize=9, handlelength=2.5)


# =========================
# Maps: points loader + dynamic extent logic
# =========================

def load_points(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None

    df = pd.read_csv(p)

    required = ["lon", "lat", "year", "source", "basin"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Points CSV missing {missing}. Have: {list(df.columns)[:60]}")

    df["source"] = df["source"].astype(str).str.upper().str.strip()
    df["basin"] = df["basin"].astype(str).str.strip().str.upper()

    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # period column may or may not exist in rebuilt files
    if "period" in df.columns:
        df["period"] = df["period"].astype(str).str.strip()
    else:
        df["period"] = "ALL"

    def _p_short(row) -> str:
        p = str(row["period"]).strip()

        if p.startswith("20c") or p == "20c":
            return "20c"
        if p.startswith("21c") or p == "21c":
            return "21c"

        # fallback for ALL files: infer from year
        y = row["year"]
        if pd.notna(y):
            y = int(y)
            if 1950 <= y <= 1999:
                return "20c"
            if 2000 <= y <= 2025:
                return "21c"

        return ""

    df["period_short"] = df.apply(_p_short, axis=1)

    df["src_group"] = np.where(
        df["source"] == "ARGO", "ARGO",
        np.where(df["source"] == "SDN", "SDN", "ADRIATIC")
    )

    df = df[
        np.isfinite(df["lon"]) &
        np.isfinite(df["lat"]) &
        df["period_short"].isin(["20c", "21c"])
    ].copy()

    return df


def _np_coords_from_basin_obj(bobj: Any) -> Optional[np.ndarray]:
    """Try several attribute names to extract Nx2 lon/lat coordinates from a Basin/ComposedBasin object."""
    candidates = []
    for attr in ("points", "poly", "polygon", "coords", "array", "vertices"):
        if hasattr(bobj, attr):
            candidates.append(getattr(bobj, attr))
    # Some objects might have a method
    for meth in ("to_array", "as_array", "get_points"):
        if hasattr(bobj, meth) and callable(getattr(bobj, meth)):
            try:
                candidates.append(getattr(bobj, meth)())
            except Exception:
                pass

    for c in candidates:
        try:
            arr = np.asarray(c)
            if arr.ndim == 2 and arr.shape[0] >= 3:
                # allow Nx3 with [lon,lat,0]
                if arr.shape[1] >= 2:
                    return arr[:, :2].astype(float)
        except Exception:
            continue
    return None


def _extent_from_coords(coords: np.ndarray, pad: float) -> Optional[Tuple[float, float, float, float]]:
    if coords is None or len(coords) < 3:
        return None
    lon = coords[:, 0]
    lat = coords[:, 1]
    if not (np.isfinite(lon).any() and np.isfinite(lat).any()):
        return None
    lonmin, lonmax = float(np.nanmin(lon)), float(np.nanmax(lon))
    latmin, latmax = float(np.nanmin(lat)), float(np.nanmax(lat))
    return (lonmin - pad, lonmax + pad, latmin - pad, latmax + pad)


def _extent_from_points(pts: Optional[pd.DataFrame], basin: str, pad: float) -> Optional[Tuple[float, float, float, float]]:
    if pts is None or pts.empty:
        return None
    p = pts[pts["basin"] == str(basin).upper()]
    if p.empty:
        return None
    lon = p["lon"].to_numpy(float)
    lat = p["lat"].to_numpy(float)
    if lon.size == 0 or lat.size == 0:
        return None
    lonmin, lonmax = float(np.nanmin(lon)), float(np.nanmax(lon))
    latmin, latmax = float(np.nanmin(lat)), float(np.nanmax(lat))
    return (lonmin - pad, lonmax + pad, latmin - pad, latmax + pad)


def _extent_from_basins_module(basin: str, pad: float) -> Optional[Tuple[float, float, float, float]]:
    """Try Basins.manca.<name> where name is lower-case code (df1, dj3, west, ion, ...).
    For composed basins, if coords are not directly available, try to combine member extents.
    """
    try:
        import Basins  # type: ignore
    except Exception:
        return None

    if not hasattr(Basins, "manca"):
        return None

    name = str(basin).strip().lower()
    if not hasattr(Basins.manca, name):
        return None

    bobj = getattr(Basins.manca, name)

    coords = _np_coords_from_basin_obj(bobj)
    ext = _extent_from_coords(coords, pad) if coords is not None else None
    if ext is not None:
        return ext

    # composed basin fallback: union member extents if members available
    for attr in ("members", "basins", "children", "components"):
        if hasattr(bobj, attr):
            try:
                members = getattr(bobj, attr)
                parts = []
                for m in members:
                    cc = _np_coords_from_basin_obj(m)
                    ee = _extent_from_coords(cc, pad) if cc is not None else None
                    if ee is not None:
                        parts.append(ee)
                if parts:
                    lonmin = min(e[0] for e in parts)
                    lonmax = max(e[1] for e in parts)
                    latmin = min(e[2] for e in parts)
                    latmax = max(e[3] for e in parts)
                    return (lonmin, lonmax, latmin, latmax)
            except Exception:
                pass

    return None


def get_extent(basin: str, pts: Optional[pd.DataFrame]) -> Tuple[float, float, float, float]:
    b = str(basin).upper()
    if b in EXTENT_OVERRIDE and EXTENT_OVERRIDE[b]:
        return EXTENT_OVERRIDE[b]

    ext = _extent_from_basins_module(b, EXTENT_PAD_DEG)
    if ext is not None:
        return ext

    ext = _extent_from_points(pts, b, EXTENT_PAD_DEG)
    if ext is not None:
        return ext

    return EXTENT_DEFAULT


def plot_map(ax, pts: Optional[pd.DataFrame], basin: str, period_short: str, title: str):
    if not HAVE_CARTOPY:
        ax.text(0.5, 0.5, "Cartopy not installed", ha="center", va="center")
        ax.axis("off")
        return

    proj = ccrs.PlateCarree()
    pos = ax.get_position()
    ax.remove()
    axp = plt.gcf().add_axes(pos, projection=proj)
    axp.set_aspect("auto")
    axp.set_anchor("N")

    axp.set_title(title, fontsize=11)
    axp.add_feature(cfeature.LAND, facecolor="0.95", zorder=0)
    axp.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
    axp.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5, zorder=1)
    axp.gridlines(draw_labels=False, linewidth=0.3, alpha=0.35)

    lonmin, lonmax, latmin, latmax = get_extent(basin, pts)
    axp.set_extent([lonmin, lonmax, latmin, latmax], crs=proj)

    if pts is None or pts.empty:
        axp.text(0.5, 0.5, "No points CSV provided", ha="center", va="center", transform=axp.transAxes)
        return

    p = pts[(pts["basin"] == str(basin).upper()) & (pts["period_short"] == period_short)].copy()
    if p.empty:
        axp.text(0.5, 0.5, f"No points for {period_short}", ha="center", va="center", transform=axp.transAxes)
        return

    styles = {
        "ARGO":     dict(marker="o", s=10, facecolor="#8EC5E8", edgecolor="none", alpha=0.75, zorder=3),
        "SDN":      dict(marker="o", s=10, facecolor="k",        edgecolor="none", alpha=0.35, zorder=3),
        "ADRIATIC": dict(marker="^", s=16, facecolor="#4A8F6B", edgecolor="k",    alpha=0.80, linewidth=0.3, zorder=4),
    }

    for grp in ["SDN", "ADRIATIC", "ARGO"]:
        gg = p[p["src_group"] == grp]
        if gg.empty:
            continue
        st = styles[grp]
        axp.scatter(gg["lon"], gg["lat"], transform=proj, **st)

    import matplotlib.lines as mlines
    import matplotlib.lines as mlines
    handles = []

    # always show SDN if present
    if (p["src_group"] == "SDN").any():
        handles.append(
            mlines.Line2D([], [], marker=styles["SDN"]["marker"], linestyle="None",
                          markerfacecolor=styles["SDN"]["facecolor"], markeredgecolor="none",
                          markersize=6, label="SDN")
        )

    # show In-situ ONLY for Adriatic basins AND only if present
    if is_adriatic_basin(basin) and (p["src_group"] == "ADRIATIC").any():
        handles.append(
            mlines.Line2D([], [], marker=styles["ADRIATIC"]["marker"], linestyle="None",
                          markerfacecolor=styles["ADRIATIC"]["facecolor"], markeredgecolor="k",
                          markersize=7, label="In-situ")
        )

    # always show Argo if present
    if (p["src_group"] == "ARGO").any():
        handles.append(
            mlines.Line2D([], [], marker=styles["ARGO"]["marker"], linestyle="None",
                          markerfacecolor=styles["ARGO"]["facecolor"], markeredgecolor="none",
                          markersize=6, label="Argo")
        )

    if handles:
        axp.legend(handles=handles, loc="lower left", frameon=False, fontsize=8, borderpad=0.2, handletextpad=0.4)

# =========================
# Insets (unchanged)
# =========================

def add_inset_profiles(parent_ax, df: pd.DataFrame, var: str, std: str, zmax_inset: float = 200.0,
                       bbox=(0.55, 0.12, 0.40, 0.40), linestyle: str = "-", basin: Optional[str] = None,
                       var_key: Optional[str] = None, row_key: Optional[str] = None):
    """Add a 0–zmax_inset inset for a profile axis.

    bbox can be:
      - a 4-tuple (x0, y0, w, h) in axis-fraction coordinates, OR
      - a callable like get_inset_bbox(basin, var_key, row_key).
    """
    bb = bbox
    if callable(bb):
        bb = bb(basin, var_key, row_key)
    iax = parent_ax.inset_axes(bb)
    plot_profile(iax, df, var, std, float(zmax_inset), title="", xlabel="", show_title=False, linestyle=linestyle, basin=basin)
    iax.set_ylabel("")
    iax.set_xlabel("")
    iax.set_title(f"0–{int(zmax_inset)} m", fontsize=9)
    iax.tick_params(labelsize=8)
    iax.grid(True, alpha=0.20)
    return iax


def add_inset_delta(parent_ax, df: pd.DataFrame, var: str, std: str, zmax_inset: float = 200.0,
                    bbox=(0.55, 0.12, 0.40, 0.40), basin: Optional[str] = None,
                    var_key: Optional[str] = None, row_key: Optional[str] = None):
    bb = bbox
    if callable(bb):
        bb = bb(basin, var_key, row_key)
    iax = parent_ax.inset_axes(bb)
    plot_delta(iax, df, var, std, float(zmax_inset), xlabel="", basin=basin)
    iax.set_ylabel("")
    iax.set_xlabel("")
    iax.set_title(f"0–{int(zmax_inset)} m", fontsize=9)
    iax.tick_params(labelsize=8)
    iax.grid(True, alpha=0.20)
    return iax


def build(df21: pd.DataFrame, df20: pd.DataFrame, basin: str, pts: Optional[pd.DataFrame], out_png: str):
    basin = str(basin).upper().strip()

    zmax = zmax_valid_for_basin(df21, df20, basin)
    d21 = df21[df21["BASIN"] == basin].copy()
    d20 = df20[df20["BASIN"] == basin].copy()
    dlt = compute_delta(df21, df20, basin)

    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.45], wspace=0.26, hspace=0.30)

    axT21 = fig.add_subplot(gs[0, 0]); axS21 = fig.add_subplot(gs[0, 1]); axP21 = fig.add_subplot(gs[0, 2]); axMap21 = fig.add_subplot(gs[0, 3])
    axT20 = fig.add_subplot(gs[1, 0]); axS20 = fig.add_subplot(gs[1, 1]); axP20 = fig.add_subplot(gs[1, 2]); axMap20 = fig.add_subplot(gs[1, 3])
    axTd  = fig.add_subplot(gs[2, 0]); axSd  = fig.add_subplot(gs[2, 1]); axPd  = fig.add_subplot(gs[2, 2]); axSup  = fig.add_subplot(gs[2, 3])

    fig.suptitle(f"{basin}: Seasonal climatology — 21st vs 20th + Δ (+ PDA)", fontsize=16, y=0.98)

    plot_profile(axT21, d21, "T_mean", "T_std", zmax, "Temperature [°C]", "T [°C]", linestyle=LINESTYLE_21C, basin=basin)
    plot_profile(axS21, d21, "S_mean", "S_std", zmax, "Salinity [-]", "S [-]", linestyle=LINESTYLE_21C, basin=basin)
    plot_profile(axP21, d21, "SIG0_mean", "SIG0_std", zmax, "PDA: σ₀ [kg m⁻³]", "σ₀ [kg m⁻³]", linestyle=LINESTYLE_21C, basin=basin)

    plot_profile(axT20, d20, "T_mean", "T_std", zmax, "", "T [°C]", show_title=False, linestyle=LINESTYLE_20C, basin=basin)
    plot_profile(axS20, d20, "S_mean", "S_std", zmax, "", "S [-]", show_title=False, linestyle=LINESTYLE_20C, basin=basin)
    plot_profile(axP20, d20, "SIG0_mean", "SIG0_std", zmax, "", "σ₀ [kg m⁻³]", show_title=False, linestyle=LINESTYLE_20C, basin=basin)

    _unify_xlim(axT21, axT20)
    _unify_xlim(axS21, axS20)
    _unify_xlim(axP21, axP20)

    plot_delta(axTd, dlt, "dT", "dT_std", zmax, "ΔT [°C]", basin=basin)
    plot_delta(axSd, dlt, "dS", "dS_std", zmax, "ΔS [-]", basin=basin)
    plot_delta(axPd, dlt, "dSIG0", "dSIG0_std", zmax, "Δσ₀ [kg m⁻³]", basin=basin)
    axPd.set_title("")

    if float(zmax) > 200 and (str(basin).upper() not in NO_INSET_BASINS):
        add_inset_profiles(
            axT21, d21, "T_mean", "T_std", 200.0,
            bbox=get_inset_bbox(basin, "T", "21c"),
            linestyle=LINESTYLE_21C, basin=basin
        )
        add_inset_profiles(
            axS21, d21, "S_mean", "S_std", 200.0,
            bbox=get_inset_bbox(basin, "S", "21c"),
            linestyle=LINESTYLE_21C, basin=basin
        )
        add_inset_profiles(
            axP21, d21, "SIG0_mean", "SIG0_std", 200.0,
            bbox=get_inset_bbox(basin, "SIG0", "21c"),
            linestyle=LINESTYLE_21C, basin=basin
        )

        add_inset_profiles(
            axT20, d20, "T_mean", "T_std", 200.0,
            bbox=get_inset_bbox(basin, "T", "20c"),
            linestyle=LINESTYLE_20C, basin=basin
        )
        add_inset_profiles(
            axS20, d20, "S_mean", "S_std", 200.0,
            bbox=get_inset_bbox(basin, "S", "20c"),
            linestyle=LINESTYLE_20C, basin=basin
        )
        add_inset_profiles(
            axP20, d20, "SIG0_mean", "SIG0_std", 200.0,
            bbox=get_inset_bbox(basin, "SIG0", "20c"),
            linestyle=LINESTYLE_20C, basin=basin
        )

        add_inset_delta(
            axTd, dlt, "dT", "dT_std", 200.0,
            bbox=get_inset_bbox(basin, "dT", "dlt"),
            basin=basin
        )
        add_inset_delta(
            axSd, dlt, "dS", "dS_std", 200.0,
            bbox=get_inset_bbox(basin, "dS", "dlt"),
            basin=basin
        )
        add_inset_delta(
            axPd, dlt, "dSIG0", "dSIG0_std", 200.0,
            bbox=get_inset_bbox(basin, "dSIG0", "dlt"),
            basin=basin
        )

    axT21.set_ylabel("21st c. (2000–2025)\nDepth [m]")
    axT20.set_ylabel("20th c. (1950–1999)\nDepth [m]")
    axTd.set_ylabel("Δ (21c − 20c)\nDepth [m]")

    fig.subplots_adjust(bottom=0.10, top=0.92)

    plot_map(axMap21, pts, basin, "21c", f"{basin} | 21c_2000_2025")
    plot_map(axMap20, pts, basin, "20c", f"{basin} | 20c_1950_1999")

    plot_support(axSup, COUNTS_CSV, basin, zmax)
    axSup.set_title("Sampling support")

    import matplotlib.lines as mlines
    season_handles = [mlines.Line2D([], [], color=SEASON_COLOR[s], lw=2, label=SEASON_LABEL[s]) for s in SEASON_ORDER]

    left = axTd.get_position().x0
    right = axPd.get_position().x1
    center = 0.5 * (left + right)

    fig.legend(
        handles=season_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(center, 0.02),
        fontsize=10
    )

    outp = Path(out_png)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=200)
    plt.close(fig)
    print("Saved:", outp)


# =========================
# Driver
# =========================

def auto_basin_list(df21: pd.DataFrame, df20: pd.DataFrame) -> list[str]:
    b = set(df21["BASIN"].astype(str).str.upper().unique()) | set(df20["BASIN"].astype(str).str.upper().unique())
    return sorted([x for x in b if x and x != "NAN"])

def run_mode(mode: str, basins: Optional[list[str]] = None):
    mode = mode.lower().strip()

    if mode == "atomic":
        df21 = load_climatology(CLIM_21C_ATOMIC_CSV)
        df20 = load_climatology(CLIM_20C_ATOMIC_CSV)
        pts = load_points(POINTS_META_ATOMIC_CSV)
        mode_dir = "atomic"

    elif mode == "composed":
        df21 = load_climatology(CLIM_21C_COMPOSED_CSV)
        df20 = load_climatology(CLIM_20C_COMPOSED_CSV)
        pts = load_points(POINTS_META_COMPOSED_CSV)
        mode_dir = "composed"

    else:
        raise ValueError("mode must be atomic or composed")

    if basins is None or len(basins) == 0:
        basins_to_plot = auto_basin_list(df21, df20)
    else:
        basins_to_plot = [b.upper() for b in basins]

    outdir = Path(OUTDIR) / mode_dir
    outdir.mkdir(parents=True, exist_ok=True)

    for basin in basins_to_plot:
        out_png = outdir / f"{basin}_{mode_dir}_publication_3row_PDA_maps_support.png"
        build(df21, df20, basin, pts, str(out_png))
        
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["atomic", "composed", "both"], default="both",
                    help="Which climatology set to plot.")
    ap.add_argument("--basins", nargs="*", default=None,
                    help="Optional explicit basins (e.g. DJ1 DJ3 DF1 ion west). If omitted, plots all present in CSV.")
    args = ap.parse_args()

    if args.mode in ("atomic", "both"):
        run_mode("atomic", args.basins)
    if args.mode in ("composed", "both"):
        run_mode("composed", args.basins)

if __name__ == "__main__":
    main()