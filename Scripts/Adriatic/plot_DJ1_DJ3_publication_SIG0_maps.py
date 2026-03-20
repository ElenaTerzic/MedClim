#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_DJ1_DJ3_publication_SIG0_maps_support.py

Publication-ready DJ1/DJ3 figure matching the original Med layout:
- 3 rows: 21c, 20c, Δ (21c−20c)
- 4 columns: Temperature, Salinity, PDA (σ0), Maps/Support
- PDA uses SIG0_mean/SIG0_std from climatology CSVs (produced by TS+SIG0 pipeline)
- Two maps: row1 col4 = 21c, row2 col4 = 20c (ARGO colored by year; in-situ markers)
- Support panel: row3 col4, with centered legend below (no overlap)
- NO dashed lines anywhere for T/S/σ0/Δ (solid lines only)

You only need to edit the USER CONFIG block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import os
import pandas as pd
from pandas.errors import EmptyDataError
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False

# =========================
# USER CONFIG
# =========================
CLIM_21C_CSV = "/Users/elena/Projects/IRB/MedClim/Stats/MultiSource_Climatology_BINNED_ELIG/21c_2000_2019/POOLED_RAW/climatology_POOLED_RAW_21c_2000_2019_atomic_after_sigma.csv"
CLIM_20C_CSV = "/Users/elena/Projects/IRB/MedClim/Stats/MultiSource_Climatology_BINNED_ELIG/20c_1950_1999/POOLED_RAW/climatology_POOLED_RAW_20c_1950_1999_atomic_after_sigma.csv"
COUNTS_CSV   = "/Users/elena/Projects/IRB/MedClim/Stats/cast_counts_bybin_by_platform_season_period_T.csv"

# Plotting-only sanity masks / overrides
SALINITY_PLOT_RANGE = (30.0, 40.5)  # [psu]
ZMAX_OVERRIDE = {"DJ1": 100.0}  # plotting cap per basin (optional)
FULL_DEPTH_BASINS = {"DJ3"}  # force full-depth axis even if deep bins are NaN
# Some climatology CSVs carry *low coverage* flags that are too aggressive for deep basins
# (they can mask the full-depth signal entirely). Disable this mask for basins listed here.
DISABLE_LOWCOV_MASK_BASINS = {"DJ3"}
# Break line connections across large depth gaps (prevents DJ1 sparse-bin artifacts).
# For deep basins (e.g., DJ3) this can unintentionally hide deep segments because standard levels are widely spaced.
GAP_BREAK_BASINS = {"DJ1"}
GAP_M = 40.0

POINTS_META_CSV = "/Users/elena/Projects/IRB/MedClim/Stats/points_metadata_all_sources_ALL.csv"  # produced by export_points_metadata_from_cast_index_STANDALONE_v2.py  # set None to disable maps points
OUTDIR = "/Users/elena/Projects/IRB/MedClim/Plots/Adriatic/Custom"

BASINS_TO_PLOT = ["DJ1", "DJ3"]

# Fallback extents (lon_min, lon_max, lat_min, lat_max)
EXTENT_OVERRIDE = {
    # Force extent even when points exist. Keys are basin abbrev (upper).
    # Example: "DJ3": (13.8, 20.6, 38.0, 44.0),
}

EXTENT_FALLBACK = {
    "DJ1": (10.8, 16.9, 43.0, 46.3),
    "DJ3": (14.4, 20.6, 39.5, 43.6),
}

# seasons/colors (match Med)
SEASON_ORDER = ["JFM", "AMJ", "JAS", "OND"]
SEASON_LABEL = {"JFM":"Winter", "AMJ":"Spring", "JAS":"Summer", "OND":"Autumn"}
SEASON_COLOR = {"JFM":"C0", "AMJ":"C2", "JAS":"C3", "OND":"C1"}


# Inset placement control (figure-relative within each axis).
# Keys: (basin, panel, period) where panel in {"T","S","SIG0","dT","dS","dSIG0"} and period in {"21c","20c","dlt"}.
# Values: (x0, y0, w, h) in axis fraction coordinates.
INSET_BBOX = {
    # DJ3: put S and SIG0 insets bottom-left for 21c/20c; keep others default (bottom-right).
    ("DJ3","S","21c"):   (0.16, 0.12, 0.40, 0.40),
    ("DJ3","S","20c"):   (0.16, 0.12, 0.40, 0.40),
    ("DJ3","SIG0","21c"):(0.16, 0.12, 0.40, 0.40),
    ("DJ3","SIG0","20c"):(0.16, 0.12, 0.40, 0.40),
    # Δ panels: keep default unless overridden
}

# Δσ0 x-limits (set to None to autoscale)
DELTA_SIG0_XLIM = (-1.0, 1.0)

PLATFORM_NORMALIZE = {
    "SDN": "SDN",
    "ARGO": "ARGO",
    "CTD_HR_2019_2023_MFSD": "ADRIATIC",
    "RB_Northern_Adriatic_Transect_1979_2018": "ADRIATIC",
    "Palagruza_Transect_1952_2018": "ADRIATIC",
    "IOR_Adriatic_database_1987_2019": "ADRIATIC",
}


MIN_N = 10  # keep same as original

# Apply MIN_N only to selected shallow basins (DJ1). Deep basins (e.g. DJ3) often
# have sparse deep-bin sampling; the publication layout still shows full-depth curves.
APPLY_MIN_N_BASINS = {"DJ1"}  # set of basin codes (upper)

def apply_min_n_mask(d: pd.DataFrame, basin: str) -> pd.DataFrame:
    """Return a copy of d with MIN_N masking applied only if basin is in APPLY_MIN_N_BASINS."""
    b = str(basin).upper()
    if MIN_N is None or MIN_N <= 1 or b not in APPLY_MIN_N_BASINS:
        return d
    out = d.copy()
    # Temperature
    if "n_T" in out.columns:
        bad = pd.to_numeric(out["n_T"], errors="coerce") < MIN_N
        out.loc[bad.fillna(False), ["T_mean", "T_std"]] = np.nan
        # σ0 fallback on n_T if no n_SIG0
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


# Line styles to distinguish periods (requested)
LINESTYLE_21C = "-"
SUPPORT_LINESTYLE_21C = "-"
SUPPORT_LINESTYLE_20C = "--"
LINESTYLE_20C = "-"  # keep profiles solid; use dashed only in support panel
# =========================


def _must_have_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Have: {list(df.columns)[:50]}")


def load_climatology(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalize for common variants
    if "basin" in df.columns and "BASIN" not in df.columns:
        df.rename(columns={"basin":"BASIN"}, inplace=True)

    req = ["BASIN", "season", "z_m", "T_mean", "T_std", "S_mean", "S_std"]
    _must_have_cols(df, req)

    if "sig0_mean" in df.columns and "SIG0_mean" not in df.columns:
        df.rename(columns={"sig0_mean":"SIG0_mean", "sig0_std":"SIG0_std"}, inplace=True)

    _must_have_cols(df, ["SIG0_mean", "SIG0_std"])
    df["season"] = df["season"].astype(str)
    df["BASIN"] = df["BASIN"].astype(str)
    df["z_m"] = pd.to_numeric(df["z_m"], errors="coerce")

    return df



def zmax_valid_for_basin(df21: pd.DataFrame, df20: pd.DataFrame, basin: str) -> float:
    d21 = apply_min_n_mask(df21[df21["BASIN"] == basin].copy(), basin)
    d20 = apply_min_n_mask(df20[df20["BASIN"] == basin].copy(), basin)

    # For selected deep basins, keep the full-depth axis (matches publication layout):
    # the main panels show the entire water column; the inset provides the 0–200 m detail.
    if str(basin).upper() in FULL_DEPTH_BASINS:
        z_all = pd.concat([d21["z_m"], d20["z_m"]], ignore_index=True).dropna()
        return float(z_all.max()) if len(z_all) else 0.0

    def _valid(d):
        ok = np.isfinite(d["T_mean"]) | np.isfinite(d["S_mean"]) | np.isfinite(d["SIG0_mean"])
        return d.loc[ok, "z_m"]

    z = pd.concat([_valid(d21), _valid(d20)], ignore_index=True).dropna()
    if len(z) == 0:
        zmax_valid = float(pd.concat([d21["z_m"], d20["z_m"]], ignore_index=True).max())
    else:
        zmax_valid = float(z.max())

    zov = ZMAX_OVERRIDE.get(str(basin).upper())
    if zov is not None:
        zmax_valid = float(min(zmax_valid, zov))
    return zmax_valid



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
    """Insert NaNs after large depth gaps so matplotlib doesn't draw long connecting segments."""
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

def plot_profile(
    ax,
    df: pd.DataFrame,
    var: str,
    std: str,
    zmax: float,
    title: str,
    xlabel: str,
    show_title: bool = True,
    linestyle: str = "-",
    basin: Optional[str] = None,
):
    for season in SEASON_ORDER:
        d = df[df["season"] == season].sort_values("z_m")
        d = d[d["z_m"] <= zmax]

        # Low-coverage mask (if present in climatology CSV)
        # NOTE: for some deep basins these flags are overly strict and remove the whole deep column.
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

        # --- line (requires finite x,y) ---
        m_line = np.isfinite(x0) & np.isfinite(y0)
        if not m_line.any():
            continue
        xL, yL = x0[m_line], y0[m_line]
        if (basin is None) or (str(basin).upper() in GAP_BREAK_BASINS):
            yL, xL = _insert_nan_gaps(yL, xL, gap_m=GAP_M)

        ax.plot(xL, yL, lw=2.0, color=SEASON_COLOR[season], linestyle=linestyle)

        # --- envelope (requires finite x,y,std) ---
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


def plot_delta(ax, df: pd.DataFrame, var: str, std: str, zmax: float, xlabel: str, linestyle: str = "-", basin: Optional[str] = None):
    for season in SEASON_ORDER:
        d = df[df["season"] == season].sort_values("z_m")
        d = d[d["z_m"] <= zmax]
        if d.empty or (var not in d.columns):
            continue

        x0 = d[var].to_numpy(dtype=float)
        y0 = d["z_m"].to_numpy(dtype=float)

        # --- line ---
        m_line = np.isfinite(x0) & np.isfinite(y0)
        if not m_line.any():
            continue
        xL, yL = x0[m_line], y0[m_line]
        if (basin is None) or (str(basin).upper() in GAP_BREAK_BASINS):
            yL, xL = _insert_nan_gaps(yL, xL, gap_m=GAP_M)

        ax.plot(xL, yL, lw=2.0, color=SEASON_COLOR[season], linestyle=linestyle)

        # --- envelope ---
        if std in d.columns:
            s0 = d[std].to_numpy(dtype=float)
            m_fill = m_line & np.isfinite(s0)
            if m_fill.any():
                xF, yF, sF = x0[m_fill], y0[m_fill], s0[m_fill]
                if (basin is None) or (str(basin).upper() in GAP_BREAK_BASINS):
                    yF, xF, sF = _insert_nan_gaps(yF, xF, sF, gap_m=GAP_M)
                ax.fill_betweenx(yF, xF - sF, xF + sF, color=SEASON_COLOR[season], alpha=0.15, linewidth=0)

    # Optional tighter limits for Δσ₀ to avoid overly wide autoscaling
    if var == "dSIG0" and DELTA_SIG0_XLIM is not None:
        ax.set_xlim(*DELTA_SIG0_XLIM)

    ax.axvline(0, color="k", lw=1, alpha=0.75)
    ax.set_ylim(zmax, 0)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(xlabel)


def src_color(src: str) -> str:
    src = src.upper()
    if src == "ARGO": return "C1"   # orange
    if src == "SDN": return "C0"    # blue
    if src == "ADRIATIC": return "C4"  # purple
    return "0.3"


def normalize_counts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "basin" in d.columns and "BASIN" not in d.columns:
        d.rename(columns={"basin":"BASIN"}, inplace=True)
    if "source" in d.columns:
        d["source"] = d["source"].astype(str).str.upper()
    d["PERIOD"] = d["period"].astype(str).str.replace("21c_2000_2019", "21c").str.replace("20c_1950_1999", "20c")
    d["season"] = d["season"].astype(str)
    d["z_m"] = pd.to_numeric(d["z_m"], errors="coerce")
    d["n_casts"] = pd.to_numeric(d["n_casts"], errors="coerce")
    return d


def summarize_counts(dfc: pd.DataFrame, zmax: float) -> pd.DataFrame:
    """Summarize counts CSV into per-(period, source, season) statistics.

    Input (preferred columns):
      basin, season, period, z_m, source, n_casts
    Optional:
      sdn_class (for SDN subtypes). If present, counts are FIRST summed across sdn_class
      within each depth bin before computing percentiles, so SDN isn't overweighted.

    Statistics are computed across depth bins with z_m <= zmax:
      p25, p75, median, nmax of n_casts(z)
    """
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
    d["z_m"] = pd.to_numeric(d["z_m"], errors="coerce")
    d["n_casts"] = pd.to_numeric(d["n_casts"], errors="coerce")
    if "sdn_class" in d.columns:
        d["sdn_class"] = d["sdn_class"].astype(str)

    d = d[np.isfinite(d["z_m"]) & np.isfinite(d["n_casts"])].copy()
    d = d[d["z_m"] <= float(zmax)].copy()
    if d.empty:
        return pd.DataFrame(columns=["period","source","season","p25","p75","median","nmax"])

    # SUM across SDN classes within each depth bin (if present)
    gcols = ["period", "source", "season", "z_m"]
    d = d.groupby(gcols, dropna=False, as_index=False)["n_casts"].sum()

    out = []
    for (per, src, seas), g in d.groupby(["period","source","season"], dropna=False):
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
    """Sampling support panel: casts per depth-bin by season and platform.
    - whiskers = P25–P75 of n_casts across depth bins (z <= zmax)
    - circles  = median
    - squares  = max
    - solid    = 21c, dashed = 20c
    """
    stat = None

    # Counts CSV (preferred)
    if counts_csv and Path(counts_csv).exists() and Path(counts_csv).stat().st_size > 0:
        try:
            dfc = pd.read_csv(counts_csv)
            # basin filter (accept 'basin' or 'BASIN')
            bcol = "basin" if "basin" in dfc.columns else ("BASIN" if "BASIN" in dfc.columns else None)
            if bcol is not None:
                dfc = dfc[dfc[bcol].astype(str).str.upper() == str(basin).upper()].copy()
            if len(dfc):
                stat = summarize_counts(dfc, zmax=float(zmax))
        except Exception:
            stat = None

    # If no counts available, show a clean placeholder
    if stat is None or stat.empty:
        ax.text(0.5, 0.5, "No counts CSV", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    # normalize period label
    stat["period_short"] = stat["period"].astype(str)
    stat["period_short"] = stat["period_short"].str.replace("21c_2000_2019", "21c", regex=False)
    stat["period_short"] = stat["period_short"].str.replace("20c_1950_1999", "20c", regex=False)

    # normalize source/platform label for plotting
    stat["source_norm"] = stat["source"].astype(str).map(PLATFORM_NORMALIZE).fillna(stat["source"].astype(str).str.upper())
    stat["source_norm"] = stat["source_norm"].replace({
        "ADRIATIC": "ADRIATIC",
        "ADRIATIC IN-SITU": "ADRIATIC",
        "ADRIATIC IN SITU": "ADRIATIC",
    })

    # plotting config (match Med-complete code palette)
    order = ["ARGO", "SDN", "ADRIATIC"]
    colors = {"ARGO": "C1", "SDN": "C0", "ADRIATIC": "C4"}  # orange, blue, purple
    x = np.arange(len(SEASON_ORDER))
    dx = 0.22
    dp = 0.06  # small period offset so dashed whiskers are visible

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

                # markers: filled for 21c (solid), open for 20c (dashed), same edge color
                mfc = colors[src] if per == "21c" else "none"
                mec = colors[src]
                ax.plot([xc], [row["median"]], marker="o", ms=5, mfc=mfc, mec=mec, color=mec, alpha=0.9, linestyle="None")
                ax.plot([xc], [row["nmax"]], marker="s", ms=5, mfc=mfc, mec=mec, color=mec, alpha=0.9, linestyle="None")

    ax.set_xticks(x)
    ax.set_xticklabels([SEASON_LABEL[s] for s in SEASON_ORDER], rotation=0)
    # Put y-label on the right to avoid overlap with the Δσ₀ panel
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Casts per depth-bin (n)", labelpad=8)
    ax.grid(True, alpha=0.25)

    # Legends (platform colors + period line styles)
    import matplotlib.lines as mlines
    h_src = [
        mlines.Line2D([], [], color=colors["ARGO"], lw=2, label="ARGO"),
        mlines.Line2D([], [], color=colors["SDN"], lw=2, label="SDN"),
        mlines.Line2D([], [], color=colors["ADRIATIC"], lw=2, label="ADRIATIC"),
    ]
    h_per = [
        mlines.Line2D([], [], color="k", lw=2, linestyle=SUPPORT_LINESTYLE_21C, label="21st c."),
        mlines.Line2D([], [], color="k", lw=2, linestyle=SUPPORT_LINESTYLE_20C, label="20th c."),
    ]
    # Place as two small legends to avoid clutter
    leg1 = ax.legend(handles=h_src, loc="upper right", frameon=False, fontsize=9, handlelength=2.5)
    ax.add_artist(leg1)
    ax.legend(handles=h_per, loc="upper left", frameon=False, fontsize=9, handlelength=2.5)

def load_points(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load merged points metadata (ALL/20c/21c). Expected columns at least:
    lon, lat, year, source, basin, period.

    Adds:
      - period_short: '20c' or '21c' (based on period string)
      - src_group   : 'ARGO' | 'SDN' | 'ADRIATIC' (everything else grouped as ADRIATIC)
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)

    required = ["lon", "lat", "year", "source", "basin", "period"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Points CSV missing {missing}. Have: {list(df.columns)[:40]}")

    df["source"] = df["source"].astype(str).str.upper().str.strip()
    df["basin"] = df["basin"].astype(str).str.strip()
    df["period"] = df["period"].astype(str).str.strip()

    # numeric lon/lat/year
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # period_short
    def _p_short(s: str) -> str:
        s = str(s)
        if s.startswith("20c"):
            return "20c"
        if s.startswith("21c"):
            return "21c"
        # allow already-short values
        if s in ("20c", "21c"):
            return s
        return s

    df["period_short"] = df["period"].apply(_p_short)

    # platform grouping (for map visualization)
    df["src_group"] = np.where(df["source"] == "ARGO", "ARGO",
                        np.where(df["source"] == "SDN", "SDN", "ADRIATIC"))

    # drop rows without coordinates
    df = df[np.isfinite(df["lon"]) & np.isfinite(df["lat"])].copy()
    return df


def plot_map(ax, pts: Optional[pd.DataFrame], basin: str, period_short: str, title: str):
    """Spatial distribution map for a basin & period.
    period_short: '20c' or '21c'
    Points are grouped by src_group: ARGO, SDN, ADRIATIC.
    """
    if not HAVE_CARTOPY:
        ax.text(0.5, 0.5, "Cartopy not installed", ha="center", va="center")
        ax.axis("off")
        return

    proj = ccrs.PlateCarree()
    pos = ax.get_position()
    ax.remove()
    axp = plt.gcf().add_axes(pos, projection=proj)
    axp.set_aspect("auto")      # let it fill the allocated gridspec box
    axp.set_anchor("N")         # if anything still shrinks, pin it to the top


    axp.set_title(title, fontsize=11)
    axp.add_feature(cfeature.LAND, facecolor="0.95", zorder=0)
    axp.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
    axp.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5, zorder=1)
    axp.gridlines(draw_labels=False, linewidth=0.3, alpha=0.35)

    basin_key = str(basin).strip().upper()
    if basin_key in EXTENT_OVERRIDE and EXTENT_OVERRIDE[basin_key]:
        lonmin, lonmax, latmin, latmax = EXTENT_OVERRIDE[basin_key]
    else:
        lonmin, lonmax, latmin, latmax = EXTENT_FALLBACK.get(basin_key, (0, 30, 30, 46))
    axp.set_extent([lonmin, lonmax, latmin, latmax], crs=proj)

    if pts is None or pts.empty:
        axp.text(0.5, 0.5, "No points CSV provided", ha="center", va="center", transform=axp.transAxes)
        return

    p = pts[(pts["basin"] == basin) & (pts["period_short"] == period_short)].copy()
    if p.empty:
        axp.text(0.5, 0.5, f"No points for {period_short}", ha="center", va="center", transform=axp.transAxes)
        return

    # Plain platform styling to emphasize spatial extent per platform
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
        axp.scatter(
            gg["lon"], gg["lat"],
            transform=proj,
            **st
        )

    # Small in-map legend (kept compact)
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], marker=styles["SDN"]["marker"], linestyle="None",
                      markerfacecolor=styles["SDN"]["facecolor"], markeredgecolor="none",
                      markersize=6, label="SDN"),
        mlines.Line2D([], [], marker=styles["ADRIATIC"]["marker"], linestyle="None",
                      markerfacecolor=styles["ADRIATIC"]["facecolor"], markeredgecolor="k",
                      markersize=7, label="Adriatic in-situ"),
        mlines.Line2D([], [], marker=styles["ARGO"]["marker"], linestyle="None",
                      markerfacecolor=styles["ARGO"]["facecolor"], markeredgecolor="none",
                      markersize=6, label="Argo"),
    ]
    axp.legend(handles=handles, loc="lower left", frameon=False, fontsize=8, borderpad=0.2, handletextpad=0.4)




def add_inset_profiles(parent_ax, df: pd.DataFrame, var: str, std: str, zmax_inset: float = 200.0,
                       bbox=(0.55, 0.12, 0.40, 0.40), linestyle: str = "-", basin: Optional[str] = None):
    """Add a 0–zmax_inset inset for a profile axis."""
    iax = parent_ax.inset_axes(bbox)
    plot_profile(iax, df, var, std, float(zmax_inset), title="", xlabel="", show_title=False, linestyle=linestyle, basin=basin)
    iax.set_ylabel("")
    iax.set_xlabel("")
    iax.set_title(f"0–{int(zmax_inset)} m", fontsize=9)
    iax.tick_params(labelsize=8)
    # keep inset tidy
    iax.grid(True, alpha=0.20)
    return iax


def add_inset_delta(parent_ax, df: pd.DataFrame, var: str, std: str, zmax_inset: float = 200.0,
                    bbox=(0.55, 0.12, 0.40, 0.40), basin: Optional[str] = None):
    iax = parent_ax.inset_axes(bbox)
    plot_delta(iax, df, var, std, float(zmax_inset), xlabel="", basin=basin)
    iax.set_ylabel("")
    iax.set_xlabel("")
    iax.set_title(f"0–{int(zmax_inset)} m", fontsize=9)
    iax.tick_params(labelsize=8)
    iax.grid(True, alpha=0.20)
    return iax
def build(df21: pd.DataFrame, df20: pd.DataFrame, basin: str, pts: Optional[pd.DataFrame], out_png: str):
    zmax = zmax_valid_for_basin(df21, df20, basin)
    d21 = df21[df21["BASIN"] == basin].copy()
    d20 = df20[df20["BASIN"] == basin].copy()
    dlt = compute_delta(df21, df20, basin)

    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.45], wspace=0.26, hspace=0.30)

    axT21 = fig.add_subplot(gs[0,0]); axS21 = fig.add_subplot(gs[0,1]); axP21 = fig.add_subplot(gs[0,2]); axMap21 = fig.add_subplot(gs[0,3])
    axT20 = fig.add_subplot(gs[1,0]); axS20 = fig.add_subplot(gs[1,1]); axP20 = fig.add_subplot(gs[1,2]); axMap20 = fig.add_subplot(gs[1,3])
    axTd  = fig.add_subplot(gs[2,0]); axSd  = fig.add_subplot(gs[2,1]); axPd  = fig.add_subplot(gs[2,2]); axSup  = fig.add_subplot(gs[2,3])

    fig.suptitle(f"{basin}: Seasonal climatology — 21st vs 20th + Δ (+ PDA)", fontsize=16, y=0.98)

    plot_profile(axT21, d21, "T_mean", "T_std", zmax, "Temperature [°C]", "T [°C]", linestyle=LINESTYLE_21C, basin=basin)
    plot_profile(axS21, d21, "S_mean", "S_std", zmax, "Salinity [-]", "S [-]", linestyle=LINESTYLE_21C, basin=basin)
    plot_profile(axP21, d21, "SIG0_mean", "SIG0_std", zmax, "PDA: σ₀ [kg m⁻³]", "σ₀ [kg m⁻³]", linestyle=LINESTYLE_21C, basin=basin)

    plot_profile(axT20, d20, "T_mean", "T_std", zmax, "", "T [°C]", show_title=False, linestyle=LINESTYLE_20C, basin=basin)
    plot_profile(axS20, d20, "S_mean", "S_std", zmax, "", "S [-]", show_title=False, linestyle=LINESTYLE_20C, basin=basin)
    plot_profile(axP20, d20, "SIG0_mean", "SIG0_std", zmax, "", "σ₀ [kg m⁻³]", show_title=False, linestyle=LINESTYLE_20C, basin=basin)

    xT = _unify_xlim(axT21, axT20)
    xS = _unify_xlim(axS21, axS20)
    xP = _unify_xlim(axP21, axP20)

    plot_delta(axTd, dlt, "dT", "dT_std", zmax, "ΔT [°C]", basin=basin)
    plot_delta(axSd, dlt, "dS", "dS_std", zmax, "ΔS [-]", basin=basin)
    plot_delta(axPd, dlt, "dSIG0", "dSIG0_std", zmax, "Δσ₀ [kg m⁻³]", basin=basin)
    axPd.set_title("")  # remove any title

    # Insets (0–200 m) for deep basins (e.g., DJ3), to match Med-complete layout
    if float(zmax) > 200:
        # Insets (0–200 m) for deep basins (e.g., DJ3)
        inT21 = add_inset_profiles(
            axT21, d21, "T_mean", "T_std", 200.0,
            bbox=INSET_BBOX.get((str(basin).upper(), "T", "21c"), (0.55, 0.12, 0.40, 0.40)),
            linestyle=LINESTYLE_21C, basin=basin
        )
        inS21 = add_inset_profiles(
            axS21, d21, "S_mean", "S_std", 200.0,
            bbox=INSET_BBOX.get((str(basin).upper(), "S", "21c"), (0.55, 0.12, 0.40, 0.40)),
            linestyle=LINESTYLE_21C, basin=basin
        )
        inP21 = add_inset_profiles(
            axP21, d21, "SIG0_mean", "SIG0_std", 200.0,
            bbox=INSET_BBOX.get((str(basin).upper(), "SIG0", "21c"), (0.55, 0.12, 0.40, 0.40)),
            linestyle=LINESTYLE_21C, basin=basin
        )

        inT20 = add_inset_profiles(
            axT20, d20, "T_mean", "T_std", 200.0,
            bbox=INSET_BBOX.get((str(basin).upper(), "T", "20c"), (0.55, 0.12, 0.40, 0.40)),
            linestyle=LINESTYLE_20C, basin=basin
        )
        inS20 = add_inset_profiles(
            axS20, d20, "S_mean", "S_std", 200.0,
            bbox=INSET_BBOX.get((str(basin).upper(), "S", "20c"), (0.55, 0.12, 0.40, 0.40)),
            linestyle=LINESTYLE_20C, basin=basin
        )
        inP20 = add_inset_profiles(
            axP20, d20, "SIG0_mean", "SIG0_std", 200.0,
            bbox=INSET_BBOX.get((str(basin).upper(), "SIG0", "20c"), (0.55, 0.12, 0.40, 0.40)),
            linestyle=LINESTYLE_20C, basin=basin
        )

        add_inset_delta(axTd, dlt, "dT", "dT_std", 200.0, basin=basin)
        add_inset_delta(axSd, dlt, "dS", "dS_std", 200.0, basin=basin)
        add_inset_delta(axPd, dlt, "dSIG0", "dSIG0_std", 200.0, basin=basin)

        if basin.strip().upper() == "DJ3":
            _unify_xlim(inT21, inT20)
            _unify_xlim(inS21, inS20)
            _unify_xlim(inP21, inP20)


    axT21.set_ylabel("21st c. (2000–2025)\nDepth [m]")
    axT20.set_ylabel("20th c. (1950–1999)\nDepth [m]")
    axTd.set_ylabel("Δ (21c − 20c)\nDepth [m]")

    fig.subplots_adjust(bottom=0.10, top=0.92)

    plot_map(axMap21, pts, basin, "21c", "Spatial distribution (21st c.)")
    plot_map(axMap20, pts, basin, "20c", "Spatial distribution (20th c.)")

    plot_support(axSup, COUNTS_CSV, basin, zmax)

    import matplotlib.lines as mlines
    season_handles = [mlines.Line2D([], [], color=SEASON_COLOR[s], lw=2, label=SEASON_LABEL[s]) for s in SEASON_ORDER]
  
    # --- Seasonal legend centered under first three columns only ---
    left = axTd.get_position().x0   # bottom ΔT axis (col 1)
    right = axPd.get_position().x1  # bottom Δσ0 axis (col 3)
    center = 0.5 * (left + right)

    fig.legend(
        handles=season_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(center, 0.02),  # adjust vertical position here if needed
        fontsize=10
    )




    outp = Path(out_png)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=200)
    plt.close(fig)
    print("Saved:", outp)

def main():
    df21 = load_climatology(CLIM_21C_CSV)
    df20 = load_climatology(CLIM_20C_CSV)
    pts = load_points(POINTS_META_CSV)

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    for basin in BASINS_TO_PLOT:
        out_png = outdir / f"{basin}_publication_3row_PDA_maps_support.png"
        build(df21, df20, basin, pts, str(out_png))


if __name__ == "__main__":
    main()
