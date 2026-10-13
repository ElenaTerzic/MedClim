#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reads : Results/Trends/monthly_timeseries_all_bins[_composed].csv (--mode)
# Writes: Results/Trends/Robustness/PeriodSensitivity_<mode>/robustness_period_sensitivity_{accel_profiles.png,accel_profiles.pdf,<mode>.csv}

"""
robustness_period_sensitivity_two_step.py

Robustness test for start-year sensitivity of quadratic-fit acceleration,
using the SAME two-step deseasoning logic as fit_trends_all_bins.py:

STEP 1:
    y ~ cos(12m) + sin(12m) + cos(6m) + sin(6m) + intercept

STEP 2:
    quadratic fit on deseasoned series:
        y_deseas = q2 * tau^2 + q1 * tau + q0

Acceleration:
    (2*q2) * 100   [per decade^2]

This script is meant as a robustness supplement to the main trends analysis.
"""

from __future__ import annotations

import os
import argparse
from math import erf, sqrt
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

BASE_DIR = str(RESULTS / "Trends")

PATHS = {
    "atomic": {
        "in_csv":  f"{BASE_DIR}/monthly_timeseries_all_bins.csv",
        "out_dir": f"{BASE_DIR}/Robustness/PeriodSensitivity_atomic",
    },
    "composed": {
        "in_csv":  f"{BASE_DIR}/monthly_timeseries_all_bins_composed.csv",
        "out_dir": f"{BASE_DIR}/Robustness/PeriodSensitivity_composed",
    },
}

# tested start years
START_YEARS = [1950, 1960, 1970, 1980]
END_YEAR = 2025

# minimum number of monthly values for quadratic fit
MIN_N_QUAD = 24

# Plot styling
BASINS_TO_PLOT = ["NWM", "ION", "LEV"]
VARS_TO_PLOT = ["T", "S", "SIG0"]

MAX_DEPTH_PLOT = 3000.0

# smooth only for visualization
SMOOTH_WINDOW = 3

# depth guide lines
GUIDE_DEPTHS = [100, 1000]


# legend / layout
LEGEND_Y = 0.965
TOP_ADJUST = 0.93

# x-limit handling
USE_DYNAMIC_XLIMS = True
XLIM_PAD_FRAC = 0.08
FORCE_SYMMETRIC_XLIMS = False   # True only if you want symmetric around zero
MIN_SPAN_BY_VAR = {
    "T": 0.12,
    "S": 0.03,
    "SIG0": 0.03,
}


# colors by period
PERIOD_COLORS = {
    "1950-2025": "#1f77b4",
    "1960-2025": "#ff7f0e",
    "1970-2025": "#2ca02c",
    "1980-2025": "#d62728",
}

SAVE_PNG = True
SAVE_PDF = True

# ==========================================================
# HELPERS
# ==========================================================

def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def design_seasonal(t_month: np.ndarray) -> np.ndarray:
    w12 = 2.0 * np.pi / 12.0
    w6  = 2.0 * np.pi / 6.0
    return np.column_stack([
        np.cos(w12 * t_month),
        np.sin(w12 * t_month),
        np.cos(w6  * t_month),
        np.sin(w6  * t_month),
    ])


def ols_fit(X: np.ndarray, y: np.ndarray):
    X = np.asarray(X, float)
    y = np.asarray(y, float)

    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[m, :]
    y = y[m]

    n = int(len(y))
    p = int(X.shape[1])

    if n < max(p + 1, 3):
        return None

    beta, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)
    dof = n - rank

    if dof <= 0:
        stderr = np.full(p, np.nan, dtype=float)
        pvals = np.full(p, np.nan, dtype=float)
        return beta, stderr, pvals, n, rank

    resid = y - (X @ beta)
    s2 = float(np.sum(resid**2) / dof)

    XtX_pinv = np.linalg.pinv(X.T @ X)
    cov = s2 * XtX_pinv
    diag = np.diag(cov)
    diag = np.where(diag >= 0, diag, np.nan)

    stderr = np.sqrt(diag)
    tstat = beta / stderr

    pvals = np.array(
        [2.0 * (1.0 - norm_cdf(abs(z))) if np.isfinite(z) else np.nan for z in tstat],
        dtype=float
    )

    return beta, stderr, pvals, n, rank


def build_t_month(d: pd.DataFrame) -> np.ndarray:
    ym0 = int(d["year"].iloc[0]) * 12 + int(d["month"].iloc[0])
    return (d["year"] * 12 + d["month"] - ym0).to_numpy(float)


def deseason_two_step(d: pd.DataFrame):
    y = d["value"].to_numpy(float)
    t_month = build_t_month(d)

    S = design_seasonal(t_month)
    Xs = np.column_stack([S, np.ones_like(t_month)])

    out = ols_fit(Xs, y)
    if out is None:
        return None

    beta_s, se_s, p_s, n, rank = out
    y_season = Xs @ beta_s
    y_deseas = y - y_season

    return t_month, y_deseas, (beta_s, se_s, p_s, n, rank)


def fit_quadratic_period_two_step(d0: pd.DataFrame, y0: int, y1: int, min_n: int):
    d = d0[(d0["year"] >= y0) & (d0["year"] <= y1)].copy()
    d = d.dropna(subset=["value"]).sort_values("time")

    if d.empty or len(d) < min_n:
        return None

    ds = deseason_two_step(d)
    if ds is None:
        return None

    t_month, y_deseas, season_diag = ds

    t_year = t_month / 12.0
    mid_year = 0.5 * (t_year.min() + t_year.max())
    tau = t_year - mid_year

    X = np.column_stack([tau**2, tau, np.ones_like(tau)])
    out = ols_fit(X, y_deseas)
    if out is None:
        return None

    beta, stderr, pvals, n, rank = out

    q2, q1, q0 = float(beta[0]), float(beta[1]), float(beta[2])
    q2_se = float(stderr[0]) if np.isfinite(stderr[0]) else np.nan
    q2_p  = float(pvals[0])  if np.isfinite(pvals[0]) else np.nan

    accel_per_decade2 = (2.0 * q2) * 100.0
    accel_se_per_decade2 = (2.0 * q2_se) * 100.0 if np.isfinite(q2_se) else np.nan

    return accel_per_decade2, accel_se_per_decade2, q2_p, n, rank


def load_monthly(in_csv: str) -> pd.DataFrame:
    print("[INFO] Reading:", in_csv)
    df = pd.read_csv(in_csv, parse_dates=["time"])

    needed = ["basin", "z_m", "var", "time", "year", "month", "value"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    df["basin"] = df["basin"].astype(str).str.upper().str.strip()
    df["var"]   = df["var"].astype(str).str.upper().str.strip()
    df["z_m"]   = pd.to_numeric(df["z_m"], errors="coerce")
    df["year"]  = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if "n" in df.columns:
        df["n"] = pd.to_numeric(df["n"], errors="coerce")

    df = df[np.isfinite(df["z_m"]) & np.isfinite(df["year"]) & np.isfinite(df["month"])].copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    return df


# ==========================================================
# MAIN ANALYSIS
# ==========================================================

def run_period_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    gcols = ["basin", "z_m", "var"]
    n_groups = df.groupby(gcols, sort=False).ngroups
    print(f"[INFO] Fitting groups: {n_groups}")

    for i, ((basin, z, var), d) in enumerate(df.groupby(gcols, sort=False), start=1):
        if i % 500 == 0:
            print(f"[INFO] Processed {i}/{n_groups} groups")

        for y0 in START_YEARS:
            out = fit_quadratic_period_two_step(d, y0, END_YEAR, min_n=MIN_N_QUAD)
            if out is None:
                continue

            accel, accel_se, accel_p, n, rank = out

            rows.append({
                "basin": basin,
                "z_m": float(z),
                "var": var,
                "start_year": y0,
                "end_year": END_YEAR,
                "period_label": f"{y0}-{END_YEAR}",
                "accel_per_decade2": accel,
                "accel_se_per_decade2": accel_se,
                "accel_p": accel_p,
                "n_period": n,
                "rank_period": rank,
            })

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["var", "basin", "z_m", "start_year"]).reset_index(drop=True)
    return out_df


# ==========================================================
# PLOTTING
# ==========================================================

def pretty_var_label(v: str) -> str:
    return {
        "T": "Temperature",
        "S": "Salinity",
        "SIG0": "PDA",
    }.get(v, v)


def unit_label(v: str) -> str:
    return {
        "T": "Acceleration [°C decade$^{-2}$]",
        "S": "Acceleration [decade$^{-2}$]",
        "SIG0": "Acceleration [kg m$^{-3}$ decade$^{-2}$]",
    }.get(v, "Acceleration [decade$^{-2}$]")

def compute_dynamic_xlims(plot_df: pd.DataFrame, vars_to_plot: list[str]) -> dict[str, tuple[float, float]]:
    """
    Compute one x-limit per variable across all plotted basins/periods,
    using actual min/max acceleration values plus padding.
    """
    out = {}

    for var in vars_to_plot:
        sub = plot_df[(plot_df["var"] == var) & np.isfinite(plot_df["accel_per_decade2"])].copy()

        if sub.empty:
            out[var] = (-1.0, 1.0)
            continue

        vmin = float(np.nanmin(sub["accel_per_decade2"]))
        vmax = float(np.nanmax(sub["accel_per_decade2"]))

        # enforce a minimum visible range
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


def plot_period_sensitivity(res: pd.DataFrame, out_dir: str):
    if res.empty:
        print("[WARN] No results to plot.")
        return

    plot_df = res.copy()
    plot_df = plot_df[plot_df["basin"].isin(BASINS_TO_PLOT)]
    plot_df = plot_df[plot_df["var"].isin(VARS_TO_PLOT)]
    plot_df = plot_df[np.isfinite(plot_df["accel_per_decade2"])]

    if MAX_DEPTH_PLOT is not None:
        plot_df = plot_df[plot_df["z_m"] <= MAX_DEPTH_PLOT]

    if plot_df.empty:
        raise ValueError(
            f"No rows left after filtering for basins={BASINS_TO_PLOT} "
            f"and vars={VARS_TO_PLOT}."
        )

    # dynamic x-limits by variable
    if USE_DYNAMIC_XLIMS:
        xlims_by_var = compute_dynamic_xlims(plot_df, VARS_TO_PLOT)
    else:
        xlims_by_var = XLIMS

    nrows = len(BASINS_TO_PLOT)
    ncols = len(VARS_TO_PLOT)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12.2, 13.2),
        sharey=True
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    period_order = [f"{y}-{END_YEAR}" for y in START_YEARS]

    for i, basin in enumerate(BASINS_TO_PLOT):
        for j, var in enumerate(VARS_TO_PLOT):
            ax = axes[i, j]
            sub = plot_df[(plot_df["basin"] == basin) & (plot_df["var"] == var)].copy()

            # depth guides
            for gd in GUIDE_DEPTHS:
                ax.axhline(gd, linestyle=":", linewidth=0.7, color="0.78", zorder=0)

            # zero line
            ax.axvline(0.0, linestyle="--", linewidth=0.9, color="0.45", zorder=0)

            for period in period_order:
                ss = sub[sub["period_label"] == period].sort_values("z_m").copy()
                if ss.empty:
                    continue

                # smooth only for visual clarity
                ss["accel_plot"] = (
                    ss["accel_per_decade2"]
                    .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
                    .mean()
                )

                lw = 2.5 if period == "1950-2025" else 1.8
                alpha = 1.0 if period == "1950-2025" else 0.95

                ax.plot(
                    ss["accel_plot"].to_numpy(),
                    ss["z_m"].to_numpy(),
                    color=PERIOD_COLORS.get(period, None),
                    linewidth=lw,
                    alpha=alpha,
                    label=period,
                    solid_capstyle="round"
                )

            # axes style
            ax.invert_yaxis()
            ax.set_ylim(MAX_DEPTH_PLOT, 0)
            ax.set_xlim(xlims_by_var[var])

            ax.grid(True, linestyle=":", linewidth=0.35, alpha=0.22)

            if i == 0:
                ax.set_title(pretty_var_label(var), fontsize=12.5, pad=7)

            if j == 0:
                ax.set_ylabel(f"{basin}\nDepth [m]", fontsize=12)
            else:
                ax.set_ylabel("")

            if i == nrows - 1:
                ax.set_xlabel(unit_label(var), fontsize=11)

            ax.tick_params(axis="both", labelsize=10)

            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("0.25")

    # shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, 0.948),
            fontsize=10.5,
            handlelength=2.6,
            columnspacing=1.4,
            borderaxespad=0.2
        )

    # no suptitle -> cleaner supplement style
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.07,
        top=0.905,
        wspace=0.10,
        hspace=0.10
    )

    png_path = os.path.join(out_dir, "robustness_period_sensitivity_accel_profiles.png")
    pdf_path = os.path.join(out_dir, "robustness_period_sensitivity_accel_profiles.pdf")

    if SAVE_PNG:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)
    print("[OK] Saved figure:", png_path)
    if SAVE_PDF:
        print("[OK] Saved figure:", pdf_path)

# ==========================================================
# ENTRY POINT
# ==========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["atomic", "composed"],
        default="composed",
        help="Which monthly time series to analyze."
    )
    args = ap.parse_args()

    mode = args.mode
    in_csv = PATHS[mode]["in_csv"]
    out_dir = PATHS[mode]["out_dir"]

    os.makedirs(out_dir, exist_ok=True)

    print("[INFO] Mode:", mode)
    print("[INFO] Output directory:", out_dir)

    df = load_monthly(in_csv)
    res = run_period_sensitivity(df)

    out_csv = os.path.join(out_dir, f"robustness_period_sensitivity_{mode}.csv")
    res.to_csv(out_csv, index=False)
    print("[OK] Wrote:", out_csv)

    plot_period_sensitivity(res, out_dir)
    print("[DONE]")


if __name__ == "__main__":
    main()