#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-basin diagnostic plots for trend sanity-checking using the SAME seasonal removal
logic as SL_trends_all.py:

Seasonal model (fixed periods 12 and 6 months):
  y_fit(t) = A*cos(2π t/12 + phi1) + B*cos(2π t/6 + phi2) + Y

Then:
  y_deseas = y - y_fit(t)      # anomalies (mean removed!)
  trend is fit on y_deseas

For plotting only (so it doesn't look "wrong"), we add back Y:
  y_plot = y_deseas + Y
  line_plot = line + Y

One figure per basin:
  rows = T, S, SIG0
  cols = 20c (1950–1999), 21c (2000–2025)

Input CSV expected:
  basin, z_m, var, time, year, month, <value_col>, <weight_col>
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import stats


# ==========================================================
# SETTINGS (EDIT THESE)
# ==========================================================

IN_CSV = "/Users/elena/Projects/IRB/MedClim/Results/Trends/monthly_timeseries_all_bins.csv"
OUTDIR = "/Users/elena/Projects/IRB/MedClim/Plots/MultiPlatform/Trends/TimeseriesChecks"

DEPTH_TAG  = "0-10"      # "0-10", "200", "300", "1000", "2000"
SURF_RANGE = (0.0, 10.0)
DEPTH_TOL  = 2.5

Y20_MIN, Y20_MAX = 1950, 1999
Y21_MIN, Y21_MAX = 2000, 2025

# For debugging, you can restrict:
# BASINS = ["DT3"]
BASINS: list[str] = []

VARS = ["T", "S", "SIG0"]
VAR_LABEL = {"T": "T", "S": "S", "SIG0": "PDA (σ₀)"}

MIN_MONTHS = 24

FIG_W, FIG_H = 14.5, 8.2
DPI = 200

# If True, plot deseasoned anomalies around 0 (your current behaviour).
# If False, plot deseasoned shifted back by Y (recommended for readability).
PLOT_ANOMALIES_AROUND_ZERO = False

# ==========================================================


def weighted_mean(v: np.ndarray, w: np.ndarray) -> float:
    v = np.asarray(v, float)
    w = np.asarray(w, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.sum(v[m] * w[m]) / np.sum(w[m]))


def ensure_time(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"])
    if "year" not in d.columns:
        d["year"] = d["time"].dt.year
    if "month" not in d.columns:
        d["month"] = d["time"].dt.month
    return d


def pick_weight_col(df: pd.DataFrame) -> str:
    for c in ["n", "n_casts", "nprof", "n_profiles", "count"]:
        if c in df.columns:
            return c
    raise ValueError(f"No weight column found. Columns: {list(df.columns)}")


def pick_value_col(df: pd.DataFrame) -> str:
    for c in ["value", "val", "mean"]:
        if c in df.columns:
            return c
    for c in ["T_mean", "S_mean", "SIG0_mean"]:
        if c in df.columns:
            return c
    raise ValueError(f"No value column found. Columns: {list(df.columns)}")


def pool_monthly(df_sub: pd.DataFrame, value_col: str, wcol: str) -> pd.DataFrame:
    """
    If there are multiple entries per month, pool them weighted by wcol.
    Returns: time, year, month, value, w
    """
    if df_sub.empty:
        return df_sub

    g = (
        df_sub.groupby(["time", "year", "month"], as_index=False)
              .apply(lambda x: pd.Series({
                  "value": weighted_mean(x[value_col].to_numpy(), x[wcol].to_numpy()),
                  "w": float(np.nansum(x[wcol].to_numpy()))
              }))
              .reset_index(drop=True)
    )
    return g.sort_values("time").reset_index(drop=True)


def pick_series_for_depth(df_bv: pd.DataFrame, depth_tag: str, value_col: str, wcol: str) -> pd.DataFrame:
    if depth_tag == "0-10":
        d = df_bv[(df_bv["z_m"] >= SURF_RANGE[0]) & (df_bv["z_m"] <= SURF_RANGE[1])].copy()
        return pool_monthly(d, value_col, wcol)

    target = float(depth_tag)
    z = df_bv["z_m"].to_numpy(float)
    if z.size == 0:
        return df_bv.iloc[0:0].copy()

    uniq_z = np.unique(z[np.isfinite(z)])
    if uniq_z.size == 0:
        return df_bv.iloc[0:0].copy()

    nearest = float(uniq_z[np.argmin(np.abs(uniq_z - target))])
    if abs(nearest - target) > DEPTH_TOL:
        return df_bv.iloc[0:0].copy()

    d = df_bv[np.isclose(df_bv["z_m"].to_numpy(float), nearest)].copy()
    return pool_monthly(d, value_col, wcol)


def t_months_since_start(times: pd.Series) -> np.ndarray:
    """Months since first timestamp (float), consistent with T=12 and T=6 months."""
    t0 = times.min()
    return ((times - t0) / np.timedelta64(1, "D")).to_numpy(float) / 30.4375


def seasonal_cos_model(t_months: np.ndarray, A: float, B: float, phi1: float, phi2: float, Y: float) -> np.ndarray:
    t = np.asarray(t_months, float)
    return (
        A * np.cos(2.0 * np.pi * t / 12.0 + phi1) +
        B * np.cos(2.0 * np.pi * t / 6.0  + phi2) +
        Y
    )


def deseason_like_original(times: pd.Series, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit seasonal cosine model (12 & 6 months) with phases on finite samples,
    then return FULL-LENGTH arrays aligned to input:
      y_fit_full, y_deseas_full, Y_hat
    """
    times = pd.to_datetime(times)
    y = np.asarray(y, float)

    t_full = t_months_since_start(times)
    m = np.isfinite(t_full) & np.isfinite(y)

    y_fit_full = np.full_like(y, np.nan, dtype=float)
    y_deseas_full = np.full_like(y, np.nan, dtype=float)

    if m.sum() < 12:
        return y_fit_full, y_deseas_full, np.nan

    t = t_full[m]
    yy = y[m]

    p0 = [1.0, 1.0, 0.0, 0.0, float(np.nanmean(yy))]
    bounds = ([-np.inf, -np.inf, -np.pi, -np.pi, -np.inf],
              [ np.inf,  np.inf,  np.pi,  np.pi,  np.inf])

    try:
        params, _ = curve_fit(seasonal_cos_model, t, yy, p0=p0, bounds=bounds, maxfev=20000)
        fit = seasonal_cos_model(t, *params)
        y_fit_full[m] = fit
        y_deseas_full[m] = yy - fit
        Y_hat = float(params[4])
        return y_fit_full, y_deseas_full, Y_hat
    except Exception:
        return y_fit_full, y_deseas_full, np.nan


def decimal_year(times: pd.Series) -> np.ndarray:
    times = pd.to_datetime(times)
    return times.dt.year.to_numpy(float) + (times.dt.month.to_numpy(float) - 0.5) / 12.0


def linreg_per_decade(times: pd.Series, y: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Linear regression y ~ t_years.
    Returns: slope_per_decade, intercept, p_value, r_value, stderr_per_decade
    """
    t = decimal_year(times)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    slope, intercept, r, p, stderr = stats.linregress(t[m], y[m])  # slope per year
    return float(slope * 10.0), float(intercept), float(p), float(r), float(stderr * 10.0)


def plot_basin(basin: str, df_all: pd.DataFrame, value_col: str, wcol: str):
    fig, axes = plt.subplots(3, 2, figsize=(FIG_W, FIG_H), sharex=False, sharey=False)

    periods = [("20c", Y20_MIN, Y20_MAX), ("21c", Y21_MIN, Y21_MAX)]
    colors = {"T": "tab:red", "S": "tab:blue", "SIG0": "tab:green"}

    for i, var in enumerate(VARS):
        df_bv = df_all[(df_all["basin"] == basin) & (df_all["var"] == var)].copy()
        if df_bv.empty:
            for j in range(2):
                axes[i, j].axis("off")
            continue

        ts = pick_series_for_depth(df_bv, DEPTH_TAG, value_col, wcol)
        if ts.empty:
            for j in range(2):
                ax = axes[i, j]
                ax.set_title(f"{VAR_LABEL[var]} — no data at {DEPTH_TAG}", fontsize=10)
                ax.axis("off")
            continue

        ts["time"] = pd.to_datetime(ts["time"])
        ts = ts.sort_values("time").reset_index(drop=True)

        for j, (_tag, y0, y1) in enumerate(periods):
            ax = axes[i, j]
            sub = ts[(ts["year"] >= y0) & (ts["year"] <= y1)].copy()
            if sub.empty:
                ax.axis("off")
                continue

            times = pd.to_datetime(sub["time"])
            y_raw = sub["value"].to_numpy(float)

            m_raw = np.isfinite(y_raw)
            if m_raw.sum() < MIN_MONTHS:
                ax.scatter(times[m_raw], y_raw[m_raw], s=10, alpha=0.5, color="0.5", edgecolor="none")
                ax.text(0.02, 0.98, f"too few months (n={int(m_raw.sum())})",
                        transform=ax.transAxes, va="top", ha="left", fontsize=9)
                ax.grid(True, lw=0.3, alpha=0.5)
                if i == 0:
                    ax.set_title(f"{y0}–{y1}", fontsize=11)
                if j == 0:
                    ax.set_ylabel(VAR_LABEL[var], fontsize=11)
                continue

            # seasonal fit + deseason (FULL-LENGTH)
            y_fit, y_deseas, Y_hat = deseason_like_original(times, y_raw)

            # regression on deseasoned anomalies
            slope_dec, intercept, pval, r, stderr_dec = linreg_per_decade(times, y_deseas)

            # fit line on same time grid (anomaly space)
            t_year = decimal_year(times)
            m_fit = np.isfinite(t_year) & np.isfinite(y_deseas) & np.isfinite(slope_dec) & np.isfinite(intercept)
            line = np.full_like(t_year, np.nan, dtype=float)
            if m_fit.any():
                slope_per_year = slope_dec / 10.0
                line[m_fit] = slope_per_year * t_year[m_fit] + intercept

            # --- choose what to plot
            if PLOT_ANOMALIES_AROUND_ZERO:
                y_plot = y_deseas
                line_plot = line
                ylabel = f"{VAR_LABEL[var]} (deseasoned anomaly)"
            else:
                # shift anomalies back to raw scale using fitted mean Y
                offset = Y_hat if np.isfinite(Y_hat) else float(np.nanmean(y_raw[m_raw]))
                y_plot = y_deseas + offset
                line_plot = line + offset
                ylabel = f"{VAR_LABEL[var]} (deseasoned, mean added back)"

            # --- plotting
            ax.scatter(times[m_raw], y_raw[m_raw], s=10, alpha=0.25, color="0.6",
                       edgecolor="none", label="monthly (raw)")
            ax.scatter(times, y_plot, s=14, alpha=0.85, color=colors[var],
                       edgecolor="none", label="deseasoned")
            ax.plot(times, line_plot, color="k", lw=2.0, label="linear fit")

            ax.grid(True, lw=0.3, alpha=0.5)
            ax.tick_params(labelsize=9)

            if i == 0:
                ax.set_title(f"{y0}–{y1}", fontsize=11)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=11)

            txt = (
                f"trend = {slope_dec:+.4g}/dec\n"
                f"p = {pval:.2e}\n"
                f"n = {int(np.isfinite(y_deseas).sum())}"
            )
            ax.text(
                0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9)
            )

            # one legend (top row, right panel)
            if i == 0 and j == 1:
                h, l = ax.get_legend_handles_labels()
                fig.legend(h, l, loc="upper right",
                           frameon=False, fontsize=9, bbox_to_anchor=(0.98, 0.98))

    fig.suptitle(
        f"{basin} — {DEPTH_TAG} m: raw + deseasoned (cos12/cos6) + linear trend",
        fontsize=14
    )

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"TScheck_cosfit_{basin}_{DEPTH_TAG.replace('-','_')}.png")
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    plt.savefig(out, dpi=DPI)
    plt.close(fig)
    print("Saved:", out)


def main():
    print("Reading:", IN_CSV)
    df = pd.read_csv(IN_CSV)
    df["basin"] = df["basin"].astype(str).str.upper().str.strip()
    df["var"] = df["var"].astype(str).str.upper().str.strip()
    df = ensure_time(df)

    wcol = pick_weight_col(df)
    vcol = pick_value_col(df)
    print("Using value column:", vcol)
    print("Using weight column:", wcol)

    basins = BASINS if BASINS else sorted(df["basin"].dropna().unique().tolist())
    print(f"Basins to plot: {len(basins)}")

    for b in basins:
        plot_basin(b, df, vcol, wcol)

    print("Done.")


if __name__ == "__main__":
    main()