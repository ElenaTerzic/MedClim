#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hovmöller diagrams (year x depth) for composed Mediterranean basins.

Input
-----
monthly_timeseries_all_bins_composed.csv

Expected columns
----------------
- basin
- z_m
- var        : T, S, SIG0
- time
- year
- month
- value
- n

Method
------
1. Read monthly composed-basin series
2. Aggregate monthly -> yearly using weighted mean (weights = n)
3. Build reference climatology over REF_YEARS
4. Compute yearly anomalies relative to REF_YEARS
5. Plot Hovmöller panels (T, S, SIG0) per basin
6. Also compute one overall "MED" basin from all composed basins

Notes
-----
- Uses original depth-bin centers (z_m) from the pipeline
- No interpolation by default: this lets you see the native vertical structure
- Optional rolling smoothing in time
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl
import cmocean

# ==========================================================
# USER SETTINGS
# ==========================================================

IN_CSV = "/Users/elena/Projects/IRB/MedClim/Results/Trends/monthly_timeseries_all_bins_composed.csv"
OUT_DIR = "/Users/elena/Projects/IRB/MedClim/Plots/Hovmoeller_Composed"

# Plot which basins; if None, infer from file
BASINS_TO_PLOT = None
# Example:
# BASINS_TO_PLOT = ["ALB", "WEST", "NWM", "TYR", "SIC", "ION", "LEV", "NADR", "MADR", "SADR", "MED"]

YEAR_MIN = 1950
YEAR_MAX = 2025

REF_YEAR_MIN = 1950
REF_YEAR_MAX = 1999

# Optional time smoothing after yearly aggregation
APPLY_ROLLING = True
ROLLING_YEARS = 3
ROLLING_CENTER = True

# Depth handling
MAX_DEPTH_M = 2000.0   # set e.g. 1500 if preferred
MIN_VALID_MONTHS_PER_YEAR = 1

# Masking of sparse yearly bins
MIN_N_PER_YEAR = 1.0   # yearly total n must be >= this to keep the value

# Add grey background to your plot
AX_FACE = "#f0f0f0"
FIG_FACE = "white"

# Layout
NROWS = 3
NCOLS = 1
FIGSIZE = (14, 12)

# Basin-specific depth cropping
USE_BASIN_SPECIFIC_ZMAX = True

# A depth bin is considered valid for setting basin zmax if it has at least
# this many valid years in at least one variable
MIN_VALID_YEARS_FOR_ZMAX = 5
# Figure
DPI = 300
CMAP = cmocean.cm.balance

# Variable settings
VAR_ORDER = ["T", "S", "SIG0"]
VAR_LABELS = {
    "T": "Temperature anomaly (°C)",
    "S": "Salinity anomaly",
    "SIG0": r"$\sigma_0$ anomaly (kg m$^{-3}$)",
}
DEFAULT_VAR_LIMITS = {
    "T": 1.0,
    "S": 0.20,
    "SIG0": 0.25,
}

BASIN_VAR_LIMITS = {
    "MED":  {"T": 1.0, "S": 0.20, "SIG0": 0.25},
    "WEST": {"T": 0.8, "S": 0.16, "SIG0": 0.18},
    "NWM":  {"T": 0.7, "S": 0.14, "SIG0": 0.16},
    "TYR":  {"T": 0.7, "S": 0.16, "SIG0": 0.16},
    "SIC":  {"T": 0.8, "S": 0.18, "SIG0": 0.18},
    "ION":  {"T": 1.0, "S": 0.20, "SIG0": 0.25},
    "LEV":  {"T": 1.0, "S": 0.18, "SIG0": 0.22},
    "ALB":  {"T": 0.6, "S": 0.16, "SIG0": 0.18},
    "NADR": {"T": 1.2, "S": 0.25, "SIG0": 0.20},
    "MADR": {"T": 1.2, "S": 0.25, "SIG0": 0.20},
    "SADR": {"T": 1.2, "S": 0.25, "SIG0": 0.18},
}
# If True, compute one overall MED basin by weighting all composed basins together
ADD_OVERALL_MED = True

# If True, annotate number of valid years / bins in title
SHOW_COVERAGE_INFO = False

# Increase label and font size
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.titlesize": 22,
})


# ==========================================================
# HELPERS
# ==========================================================

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(m):
        return np.nan
    return float(np.sum(values[m] * weights[m]) / np.sum(weights[m]))


def load_monthly(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])

    required = ["basin", "z_m", "var", "time", "year", "month", "value", "n"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["basin"] = df["basin"].astype(str).str.upper().str.strip()
    df["var"] = df["var"].astype(str).str.upper().str.strip()
    df["z_m"] = pd.to_numeric(df["z_m"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")

    df = df[
        df["basin"].notna() &
        df["var"].isin(VAR_ORDER) &
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
    """
    Aggregate monthly -> yearly using weighted mean(value, n).
    """
    rows = []

    gcols = ["basin", "z_m", "var", "year"]
    for key, g in dfm.groupby(gcols, sort=False):
        basin, z_m, var, year = key

        # Keep only valid monthly entries
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

    out = pd.DataFrame(rows)
    return out

def basin_specific_zmax(da: pd.DataFrame, basin: str) -> float:
    """
    Determine deepest useful depth for one basin based on anomaly availability.

    A depth is considered valid if, for at least one variable, it has
    >= MIN_VALID_YEARS_FOR_ZMAX finite yearly anomaly values.
    """
    g = da[da["basin"] == basin].copy()
    if g.empty:
        return MAX_DEPTH_M

    out_depths = []

    for z_m, gz in g.groupby("z_m"):
        valid = False
        for var, gv in gz.groupby("var"):
            n_valid_years = np.isfinite(gv["anom"]).sum()
            if n_valid_years >= MIN_VALID_YEARS_FOR_ZMAX:
                valid = True
                break
        if valid:
            out_depths.append(float(z_m))

    if not out_depths:
        return MAX_DEPTH_M

    return min(MAX_DEPTH_M, max(out_depths))

def get_var_limit(basin: str, var: str) -> float:
    basin = str(basin).upper()
    var = str(var).upper()

    if basin in BASIN_VAR_LIMITS and var in BASIN_VAR_LIMITS[basin]:
        return float(BASIN_VAR_LIMITS[basin][var])

    return float(DEFAULT_VAR_LIMITS[var])


def add_overall_med(dfy: pd.DataFrame) -> pd.DataFrame:
    """
    Build one overall 'MED' basin from all basins already present in dfy,
    using yearly weighted means across basins.
    """
    rows = []

    gcols = ["z_m", "var", "year"]
    for key, g in dfy.groupby(gcols, sort=False):
        z_m, var, year = key

        v = g["value"].to_numpy(dtype=float)
        w = g["n"].to_numpy(dtype=float)

        val = weighted_mean(v, w)
        nsum = float(np.nansum(w[np.isfinite(w) & (w > 0)]))

        if not np.isfinite(val) or not np.isfinite(nsum) or nsum <= 0:
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
    """
    Compute reference mean for each basin x z_m x var over REF_YEAR_MIN..REF_YEAR_MAX,
    weighted by yearly n.
    """
    dref = dfy[(dfy["year"] >= REF_YEAR_MIN) & (dfy["year"] <= REF_YEAR_MAX)].copy()

    rows = []
    gcols = ["basin", "z_m", "var"]
    for key, g in dref.groupby(gcols, sort=False):
        basin, z_m, var = key

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

    out_parts = []
    gcols = ["basin", "z_m", "var"]

    for key, g in da.groupby(gcols, sort=False):
        g = g.sort_values("year").copy()
        g["anom_raw"] = g["anom"]
        g["anom"] = (
            g["anom"]
            .rolling(window=ROLLING_YEARS, center=ROLLING_CENTER, min_periods=1)
            .mean()
        )
        out_parts.append(g)

    return pd.concat(out_parts, ignore_index=True)


def pivot_for_plot(da: pd.DataFrame, basin: str, var: str, zmax: Optional[float] = None) -> pd.DataFrame:
    g = da[(da["basin"] == basin) & (da["var"] == var)].copy()
    if zmax is not None:
        g = g[g["z_m"] <= zmax].copy()

    if g.empty:
        return pd.DataFrame()

    piv = g.pivot_table(index="z_m", columns="year", values="anom", aggfunc="mean")
    piv = piv.sort_index()
    piv = piv.reindex(sorted(piv.columns), axis=1)

    return piv


def centers_to_edges(x: np.ndarray) -> np.ndarray:
    """
    Convert 1D center coordinates to edges for pcolormesh.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)

    dx = np.diff(x)
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = x[:-1] + dx / 2.0
    edges[0] = x[0] - dx[0] / 2.0
    edges[-1] = x[-1] + dx[-1] / 2.0
    return edges


def nice_depth_ticks(max_depth: float) -> List[int]:
    if max_depth <= 150:
        return [0, 50, 100, 150]
    if max_depth <= 300:
        return [0, 50, 100, 200, 300]
    if max_depth <= 700:
        return [0, 100, 200, 300, 500, 700]
    if max_depth <= 1200:
        return [0, 100, 200, 400, 600, 800, 1000, 1200]
    if max_depth <= 2000:
        return [0, 200, 400, 600, 800, 1000, 1500, 2000]
    return [0, 500, 1000, 1500, 2000]

def plot_one_basin(
    da: pd.DataFrame,
    basin: str,
    out_dir: str,
):
    if USE_BASIN_SPECIFIC_ZMAX:
        zmax_plot = basin_specific_zmax(da, basin)
    else:
        zmax_plot = MAX_DEPTH_M

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE, constrained_layout=True)
    fig.patch.set_facecolor(FIG_FACE)
    axes = np.atleast_1d(axes).ravel()

    any_plotted = False

    for ax, var in zip(axes, VAR_ORDER):
        piv = pivot_for_plot(da, basin, var, zmax=zmax_plot)

        if piv.empty:
            ax.set_visible(False)
            continue

        Z = piv.to_numpy(dtype=float)
        depths = piv.index.to_numpy(dtype=float)
        years = piv.columns.to_numpy(dtype=float)

        x_edges = centers_to_edges(years)
        y_edges = centers_to_edges(depths)

        lim = get_var_limit(basin, var)
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

        ax.set_facecolor(AX_FACE)

        cmap = mpl.cm.get_cmap(CMAP).copy()
        cmap.set_bad(AX_FACE)

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
        ax.set_title(VAR_LABELS[var], fontsize=18, weight="bold")
        ax.set_ylabel("Depth (m)")
        ax.set_xlim(YEAR_MIN - 0.5, YEAR_MAX + 0.5)
        ax.set_ylim(zmax_plot, 0)

        yticks = [t for t in nice_depth_ticks(zmax_plot) if t <= zmax_plot]
        if len(yticks) >= 2:
            ax.set_yticks(yticks)

        if ax is axes[-1]:
            ax.set_xlabel("Year")
        else:
            ax.set_xlabel("")

        ax.tick_params(width=1.2, length=5)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        cbar = fig.colorbar(
            pcm,
            ax=ax,
            pad=0.02,
            fraction=0.05,   # thickness
            aspect=25        # shape
        )
        cbar.ax.tick_params(labelsize=13)
        cbar.set_label(VAR_LABELS[var], fontsize=14)

        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    #title = f"{basin}: yearly anomalies relative to {REF_YEAR_MIN}–{REF_YEAR_MAX}"
    #if APPLY_ROLLING and ROLLING_YEARS > 1:
    #    title += f" | {ROLLING_YEARS}-yr running mean"
    #title += f" | zmax={int(round(zmax_plot))} m"

    #fig.suptitle(title, fontsize=16, weight="bold")

    fig.suptitle(
        f"{basin} – Temperature, Salinity, Density anomalies\n"
        f"(reference: 1950–1999, 3-year smoothing)",
        fontsize=20,
        weight="bold"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"hovmoeller_{basin}.png")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote {out_png}")

# ==========================================================
# MAIN
# ==========================================================

def main():
    print(f"[INFO] Reading {IN_CSV}")
    dfm = load_monthly(IN_CSV)

    print("[INFO] Aggregating monthly -> yearly")
    dfy = monthly_to_yearly(dfm)

    if dfy.empty:
        raise RuntimeError("No yearly data produced.")

    if ADD_OVERALL_MED:
        print("[INFO] Building MED overall basin")
        dmed = add_overall_med(dfy)
        if not dmed.empty:
            dfy = pd.concat([dfy, dmed], ignore_index=True)

    print("[INFO] Computing reference climatology")
    dref = compute_reference(dfy)
    if dref.empty:
        raise RuntimeError("Reference climatology is empty. Check REF_YEAR_MIN/MAX and data availability.")

    print("[INFO] Computing anomalies")
    da = compute_anomalies(dfy, dref)
    if da.empty:
        raise RuntimeError("No anomalies produced.")

    if APPLY_ROLLING:
        print(f"[INFO] Applying rolling mean: {ROLLING_YEARS} years")
        da = apply_rolling(da)

    if BASINS_TO_PLOT is None:
        basins = sorted(da["basin"].dropna().astype(str).unique())
    else:
        basins = [str(b).upper() for b in BASINS_TO_PLOT]

    print("[INFO] Basins to plot:", basins)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    for basin in basins:
        plot_one_basin(da, basin, OUT_DIR)

    print("[DONE]")


if __name__ == "__main__":
    main()