#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reads : Results/Trends/Composed/{linear_trends_20c_21c,quad_trend_accel_fullperiod}_composed.csv
# Writes: Results/Trends/Robustness/ModelComparison_composed/robustness_model_comparison_{profiles.png,profiles.pdf,merged.csv}

"""
robustness_model_comparison.py

Supplementary robustness figure:
Compare depth-resolved trend estimates from

1) linear fit over 1950–1999
2) linear fit over 2000–2025
3) full-period quadratic fit (midpoint linear trend)

This tests whether the evidence for acceleration is consistent across model choices.

Input
-----
Outputs from fit_trends_all_bins.py in composed mode:
- linear_trends_20c_21c_composed.csv
- quad_trend_accel_fullperiod_composed.csv

Output
------
- robustness_model_comparison_profiles.png
- robustness_model_comparison_profiles.pdf
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Scripts.paths import RESULTS


# ==========================================================
# USER CONFIG
# ==========================================================

BASE_DIR = str(RESULTS / "Trends" / "Composed")
OUT_DIR = str(RESULTS / "Trends" / "Robustness" / "ModelComparison_composed")

LINEAR_CSV = os.path.join(BASE_DIR, "linear_trends_20c_21c_composed.csv")
QUAD_CSV   = os.path.join(BASE_DIR, "quad_trend_accel_fullperiod_composed.csv")

os.makedirs(OUT_DIR, exist_ok=True)

BASINS_TO_PLOT = ["NWM", "ION", "LEV"]
VARS_TO_PLOT = ["T", "S", "SIG0"]

MAX_DEPTH_PLOT = 3000.0

# Smoothing only for visualization
SMOOTH_WINDOW = 3

# Optional guide depths
GUIDE_DEPTHS = [100, 1000]

# dynamic x-limits
USE_DYNAMIC_XLIMS = True
XLIM_PAD_FRAC = 0.08
FORCE_SYMMETRIC_XLIMS = False

MIN_SPAN_BY_VAR = {
    "T": 0.12,
    "S": 0.04,
    "SIG0": 0.04,
}

# legend / spacing
LEGEND_Y = 0.955
TOP_ADJUST = 0.915

# Line styling
MODEL_STYLES = {
    "20c linear": {
        "color": "#1f77b4",
        "linewidth": 2.1,
        "linestyle": "-",
    },
    "21c linear": {
        "color": "#d62728",
        "linewidth": 2.1,
        "linestyle": "-",
    },
    "Quadratic midpoint": {
        "color": "0.20",
        "linewidth": 2.2,
        "linestyle": "--",
    },
}

SAVE_PNG = True
SAVE_PDF = True


# ==========================================================
# HELPERS
# ==========================================================

def pretty_var_label(v: str) -> str:
    return {
        "T": "Temperature",
        "S": "Salinity",
        "SIG0": "PDA",
    }.get(v, v)


def unit_label(v: str) -> str:
    return {
        "T": "Trend [°C decade$^{-1}$]",
        "S": "Trend [decade$^{-1}$]",
        "SIG0": "Trend [kg m$^{-3}$ decade$^{-1}$]",
    }.get(v, "Trend [decade$^{-1}$]")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(LINEAR_CSV):
        raise FileNotFoundError(f"Linear trends CSV not found: {LINEAR_CSV}")
    if not os.path.exists(QUAD_CSV):
        raise FileNotFoundError(f"Quadratic trends CSV not found: {QUAD_CSV}")

    lin = pd.read_csv(LINEAR_CSV)
    quad = pd.read_csv(QUAD_CSV)

    for df in (lin, quad):
        df["basin"] = df["basin"].astype(str).str.upper().str.strip()
        df["var"] = df["var"].astype(str).str.upper().str.strip()
        df["z_m"] = pd.to_numeric(df["z_m"], errors="coerce")

    return lin, quad


def build_plot_df(lin: pd.DataFrame, quad: pd.DataFrame) -> pd.DataFrame:
    keep_lin = [
        "basin", "z_m", "var",
        "trend20_per_decade", "trend21_per_decade",
        "trend20_p", "trend21_p",
        "n20", "n21",
    ]
    keep_quad = [
        "basin", "z_m", "var",
        "trend_mid_per_decade", "trend_mid_p",
        "accel_per_decade2", "accel_p",
        "n_full",
    ]

    lin = lin[keep_lin].copy()
    quad = quad[keep_quad].copy()

    df = pd.merge(
        lin, quad,
        on=["basin", "z_m", "var"],
        how="outer"
    )

    if MAX_DEPTH_PLOT is not None:
        df = df[df["z_m"] <= MAX_DEPTH_PLOT].copy()

    return df


def smooth_series(y: pd.Series, window: int = 3) -> pd.Series:
    return y.rolling(window=window, center=True, min_periods=1).mean()

def compute_dynamic_xlims_model(df: pd.DataFrame, vars_to_plot: list[str]) -> dict[str, tuple[float, float]]:
    """
    Compute one x-limit per variable across all plotted basins,
    considering all three displayed series:
      - trend20_per_decade
      - trend21_per_decade
      - trend_mid_per_decade
    """
    out = {}

    for var in vars_to_plot:
        sub = df[df["var"] == var].copy()
        if sub.empty:
            out[var] = (-1.0, 1.0)
            continue

        vals = np.concatenate([
            sub["trend20_per_decade"].to_numpy(dtype=float),
            sub["trend21_per_decade"].to_numpy(dtype=float),
            sub["trend_mid_per_decade"].to_numpy(dtype=float),
        ])
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            out[var] = (-1.0, 1.0)
            continue

        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))

        span = vmax - vmin
        min_span = MIN_SPAN_BY_VAR.get(var, 0.05)
        if span < min_span:
            mid = 0.5 * (vmin + vmax)
            vmin = mid - 0.5 * min_span
            vmax = mid + 0.5 * min_span
            span = min_span

        pad = span * XLIM_PAD_FRAC
        vmin -= pad
        vmax += pad

        if FORCE_SYMMETRIC_XLIMS:
            a = max(abs(vmin), abs(vmax))
            vmin, vmax = -a, a

        out[var] = (vmin, vmax)

    return out


# ==========================================================
# PLOTTING
# ==========================================================

def plot_model_comparison(df: pd.DataFrame):
    plot_df = df.copy()
    plot_df = plot_df[plot_df["basin"].isin(BASINS_TO_PLOT)]
    plot_df = plot_df[plot_df["var"].isin(VARS_TO_PLOT)]

    if plot_df.empty:
        raise ValueError("No rows left after filtering for selected basins/variables.")

    # dynamic x-limits by variable
    if USE_DYNAMIC_XLIMS:
        xlims_by_var = compute_dynamic_xlims_model(plot_df, VARS_TO_PLOT)
    else:
        xlims_by_var = XLIMS

    nrows = len(BASINS_TO_PLOT)
    ncols = len(VARS_TO_PLOT)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12.2, 13.0),
        sharey=True
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    for i, basin in enumerate(BASINS_TO_PLOT):
        for j, var in enumerate(VARS_TO_PLOT):
            ax = axes[i, j]
            sub = plot_df[(plot_df["basin"] == basin) & (plot_df["var"] == var)].copy()
            sub = sub.sort_values("z_m")

            if sub.empty:
                continue

            # guide depths
            for gd in GUIDE_DEPTHS:
                ax.axhline(gd, linestyle=":", linewidth=0.7, color="0.78", zorder=0)

            # zero line
            ax.axvline(0.0, linestyle="--", linewidth=0.9, color="0.45", zorder=0)

            # smoothed display series
            sub["trend20_plot"] = smooth_series(sub["trend20_per_decade"], SMOOTH_WINDOW)
            sub["trend21_plot"] = smooth_series(sub["trend21_per_decade"], SMOOTH_WINDOW)
            sub["trendmid_plot"] = smooth_series(sub["trend_mid_per_decade"], SMOOTH_WINDOW)

            # 20c linear
            ax.plot(
                sub["trend20_plot"].to_numpy(),
                sub["z_m"].to_numpy(),
                label="20c linear",
                color=MODEL_STYLES["20c linear"]["color"],
                linewidth=MODEL_STYLES["20c linear"]["linewidth"],
                linestyle=MODEL_STYLES["20c linear"]["linestyle"],
                solid_capstyle="round",
            )

            # 21c linear
            ax.plot(
                sub["trend21_plot"].to_numpy(),
                sub["z_m"].to_numpy(),
                label="21c linear",
                color=MODEL_STYLES["21c linear"]["color"],
                linewidth=MODEL_STYLES["21c linear"]["linewidth"],
                linestyle=MODEL_STYLES["21c linear"]["linestyle"],
                solid_capstyle="round",
            )

            # quadratic midpoint
            ax.plot(
                sub["trendmid_plot"].to_numpy(),
                sub["z_m"].to_numpy(),
                label="Quadratic midpoint",
                color=MODEL_STYLES["Quadratic midpoint"]["color"],
                linewidth=MODEL_STYLES["Quadratic midpoint"]["linewidth"],
                linestyle=MODEL_STYLES["Quadratic midpoint"]["linestyle"],
                solid_capstyle="round",
            )

            # style
            ax.invert_yaxis()
            ax.set_ylim(MAX_DEPTH_PLOT, 0)
            ax.set_xlim(xlims_by_var[var])

            ax.grid(True, linestyle=":", linewidth=0.35, alpha=0.22)

            if i == 0:
                ax.set_title(pretty_var_label(var), fontsize=12.5, pad=7)

            if j == 0:
                ax.set_ylabel(f"{basin}\nDepth [m]", fontsize=12)

            if i == nrows - 1:
                ax.set_xlabel(unit_label(var), fontsize=11)

            ax.tick_params(axis="both", labelsize=10)

            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("0.25")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, LEGEND_Y),
            fontsize=10.5,
            handlelength=2.8,
            columnspacing=1.8,
            borderaxespad=0.2
        )

    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.07,
        top=TOP_ADJUST,
        wspace=0.10,
        hspace=0.10
    )

    png_path = os.path.join(OUT_DIR, "robustness_model_comparison_profiles.png")
    pdf_path = os.path.join(OUT_DIR, "robustness_model_comparison_profiles.pdf")

    if SAVE_PNG:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)

    print("[OK] Saved figure:", png_path)
    if SAVE_PDF:
        print("[OK] Saved figure:", pdf_path)

# ==========================================================
# MAIN
# ==========================================================

def main():
    print("[INFO] Reading model outputs...")
    lin, quad = load_inputs()

    print("[INFO] Building merged plotting table...")
    df = build_plot_df(lin, quad)

    out_csv = os.path.join(OUT_DIR, "robustness_model_comparison_merged.csv")
    df.to_csv(out_csv, index=False)
    print("[OK] Wrote merged CSV:", out_csv)

    print("[INFO] Plotting...")
    plot_model_comparison(df)

    print("[DONE]")


if __name__ == "__main__":
    main()