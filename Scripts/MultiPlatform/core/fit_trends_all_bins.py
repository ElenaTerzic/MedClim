#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reads : Results/Trends/monthly_timeseries_all_bins[_composed].csv (--mode atomic|composed)
# Writes: Results/Trends[/Composed]/linear_trends_20c_21c_<mode>.csv, quad_trend_accel_fullperiod_<mode>.csv

"""
fit_trends_all_bins.py

Fit trends on monthly time series per basin × z_m × var, using a TWO-STEP approach:

STEP 1 (season removal):
  Fit season-only model with 12- and 6-month harmonics (cos/sin) + intercept:
    y = cos12 + sin12 + cos6 + sin6 + C
  Then deseason:
    y_deseas = y - y_season_fit

STEP 2 (trend fitting on y_deseas):
  Linear trends:
    20c: 1950–1999  -> slope per decade
    21c: 2000–2025  -> slope per decade

  Quadratic (full period):
    1950–2025, centered time in YEARS:
      y_deseas = q2*tau^2 + q1*tau + q0
    trend@midpoint per decade = q1*10
    acceleration per decade^2 = (2*q2)*100

Modes
-----
This script supports:
- atomic   : monthly_timeseries_all_bins.csv
- composed : monthly_timeseries_all_bins_composed.csv

Outputs are written with mode-specific filenames, e.g.:
- linear_trends_20c_21c_atomic.csv
- quad_trend_accel_fullperiod_atomic.csv
- linear_trends_20c_21c_composed.csv
- quad_trend_accel_fullperiod_composed.csv

Run examples
------------
python fit_trends_all_bins.py
python fit_trends_all_bins.py --mode atomic
python fit_trends_all_bins.py --mode composed

Notes
-----
- This uses the same two-step season-removal logic as your current script.
- p-values use normal approximation (screening-level), same spirit as before.
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
from math import erf, sqrt

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Scripts.paths import RESULTS


# =========================
# USER CONFIG
# =========================

BASE_DIR = str(RESULTS / "Trends")

PATHS = {
    "atomic": {
        "in_csv":  f"{BASE_DIR}/monthly_timeseries_all_bins.csv",
        "out_dir": f"{BASE_DIR}",
    },
    "composed": {
        "in_csv":  f"{BASE_DIR}/monthly_timeseries_all_bins_composed.csv",
        "out_dir": f"{BASE_DIR}/Composed",
    },
}

Y20_MIN, Y20_MAX = 1950, 1999
Y21_MIN, Y21_MAX = 2000, 2025
YFULL_MIN, YFULL_MAX = 1950, 2025

# Minimum months per fit
MIN_N_LINEAR = 12
MIN_N_QUAD   = 24


# =========================
# HELPERS
# =========================

def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def design_seasonal(t_month: np.ndarray) -> np.ndarray:
    """
    Seasonal design matrix columns: cos12, sin12, cos6, sin6.
    """
    w12 = 2.0 * np.pi / 12.0
    w6  = 2.0 * np.pi / 6.0
    return np.column_stack([
        np.cos(w12 * t_month),
        np.sin(w12 * t_month),
        np.cos(w6  * t_month),
        np.sin(w6  * t_month),
    ])


def ols_fit(X: np.ndarray, y: np.ndarray):
    """
    Robust OLS via lstsq + pseudo-inverse covariance.
    Works even if X is rank-deficient.
    Returns beta, stderr, pvals, n, rank.
    """
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
    """
    Integer month axis relative to first observation in this subset,
    matching your previous logic.
    """
    ym0 = int(d["year"].iloc[0]) * 12 + int(d["month"].iloc[0])
    return (d["year"] * 12 + d["month"] - ym0).to_numpy(float)


def deseason_two_step(d: pd.DataFrame):
    """
    Fit season-only model y ~ [cos12, sin12, cos6, sin6, 1], then return residuals.

    Returns
    -------
    t_month, y_deseas, season_diag
    """
    y = d["value"].to_numpy(float)
    t_month = build_t_month(d)

    S = design_seasonal(t_month)
    Xs = np.column_stack([S, np.ones_like(t_month)])  # seasonal + intercept only

    out = ols_fit(Xs, y)
    if out is None:
        return None

    beta_s, se_s, p_s, n, rank = out
    y_season = Xs @ beta_s
    y_deseas = y - y_season

    return t_month, y_deseas, (beta_s, se_s, p_s, n, rank)


def fit_linear_period_two_step(d0: pd.DataFrame, y0: int, y1: int, min_n: int):
    """
    Two-step:
    1) deseason
    2) linear regression on residuals
    """
    d = d0[(d0["year"] >= y0) & (d0["year"] <= y1)].copy()
    d = d.dropna(subset=["value"]).sort_values("time")

    if d.empty or len(d) < min_n:
        return None

    ds = deseason_two_step(d)
    if ds is None:
        return None

    t_month, y_deseas, season_diag = ds

    # linear fit on residuals: y_deseas = b*t + c
    X = np.column_stack([t_month, np.ones_like(t_month)])
    out = ols_fit(X, y_deseas)
    if out is None:
        return None

    beta, stderr, pvals, n, rank = out

    slope_per_month = float(beta[0])
    slope_se_per_month = float(stderr[0]) if np.isfinite(stderr[0]) else np.nan
    slope_p = float(pvals[0]) if np.isfinite(pvals[0]) else np.nan

    slope_per_decade = slope_per_month * 120.0
    slope_se_per_decade = slope_se_per_month * 120.0 if np.isfinite(slope_se_per_month) else np.nan

    return slope_per_decade, slope_se_per_decade, slope_p, n, rank


def fit_quadratic_full_two_step(d0: pd.DataFrame, y0: int, y1: int, min_n: int):
    """
    Two-step:
    1) deseason
    2) quadratic regression on residuals
    """
    d = d0[(d0["year"] >= y0) & (d0["year"] <= y1)].copy()
    d = d.dropna(subset=["value"]).sort_values("time")

    if d.empty or len(d) < min_n:
        return None

    ds = deseason_two_step(d)
    if ds is None:
        return None

    t_month, y_deseas, season_diag = ds

    # centered YEARS for polynomial stability
    t_year = t_month / 12.0
    mid_year = 0.5 * (t_year.min() + t_year.max())
    tau = t_year - mid_year

    # y_deseas = q2*tau^2 + q1*tau + q0
    X = np.column_stack([tau**2, tau, np.ones_like(tau)])
    out = ols_fit(X, y_deseas)
    if out is None:
        return None

    beta, stderr, pvals, n, rank = out

    q2, q1, q0 = float(beta[0]), float(beta[1]), float(beta[2])
    q2_se = float(stderr[0]) if np.isfinite(stderr[0]) else np.nan
    q1_se = float(stderr[1]) if np.isfinite(stderr[1]) else np.nan
    q2_p = float(pvals[0]) if np.isfinite(pvals[0]) else np.nan
    q1_p = float(pvals[1]) if np.isfinite(pvals[1]) else np.nan

    trend_mid_per_decade = q1 * 10.0
    trend_mid_se_per_decade = q1_se * 10.0 if np.isfinite(q1_se) else np.nan

    accel_per_decade2 = (2.0 * q2) * 100.0
    accel_se_per_decade2 = (2.0 * q2_se) * 100.0 if np.isfinite(q2_se) else np.nan

    return (
        trend_mid_per_decade, trend_mid_se_per_decade, q1_p,
        accel_per_decade2, accel_se_per_decade2, q2_p,
        n, rank
    )


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


# =========================
# MAIN
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["atomic", "composed"],
        default="atomic",
        help="Which monthly time series to fit trends on."
    )
    args = ap.parse_args()

    mode = args.mode

    in_csv = PATHS[mode]["in_csv"]
    out_dir = PATHS[mode]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("[INFO] Mode:", mode)
    print("[INFO] Output directory:", out_dir)

    df = load_monthly(in_csv)

    gcols = ["basin", "z_m", "var"]
    lin_rows = []
    quad_rows = []

    n_groups = df.groupby(gcols, sort=False).ngroups
    print(f"[INFO] Fitting groups: {n_groups}")

    for i, ((basin, z, var), d) in enumerate(df.groupby(gcols, sort=False), start=1):
        if i % 500 == 0:
            print(f"[INFO] Processed {i}/{n_groups} groups")

        out20 = fit_linear_period_two_step(d, Y20_MIN, Y20_MAX, min_n=MIN_N_LINEAR)
        out21 = fit_linear_period_two_step(d, Y21_MIN, Y21_MAX, min_n=MIN_N_LINEAR)

        if (out20 is not None) or (out21 is not None):
            r = {
                "basin": basin,
                "z_m": float(z),
                "var": var,
                "trend20_per_decade": np.nan,
                "trend20_se_per_decade": np.nan,
                "trend20_p": np.nan,
                "n20": 0,
                "rank20": np.nan,
                "trend21_per_decade": np.nan,
                "trend21_se_per_decade": np.nan,
                "trend21_p": np.nan,
                "n21": 0,
                "rank21": np.nan,
            }

            if out20 is not None:
                (
                    r["trend20_per_decade"],
                    r["trend20_se_per_decade"],
                    r["trend20_p"],
                    r["n20"],
                    r["rank20"],
                ) = out20

            if out21 is not None:
                (
                    r["trend21_per_decade"],
                    r["trend21_se_per_decade"],
                    r["trend21_p"],
                    r["n21"],
                    r["rank21"],
                ) = out21

            lin_rows.append(r)

        outq = fit_quadratic_full_two_step(d, YFULL_MIN, YFULL_MAX, min_n=MIN_N_QUAD)
        if outq is not None:
            (
                trend_mid,
                trend_mid_se,
                trend_mid_p,
                accel,
                accel_se,
                accel_p,
                nfull,
                rankfull,
            ) = outq

            quad_rows.append({
                "basin": basin,
                "z_m": float(z),
                "var": var,
                "trend_mid_per_decade": trend_mid,
                "trend_mid_se_per_decade": trend_mid_se,
                "trend_mid_p": trend_mid_p,
                "accel_per_decade2": accel,
                "accel_se_per_decade2": accel_se,
                "accel_p": accel_p,
                "n_full": nfull,
                "rank_full": rankfull,
            })

    lin_df = pd.DataFrame(lin_rows)
    quad_df = pd.DataFrame(quad_rows)

    if not lin_df.empty:
        lin_df = lin_df.sort_values(["var", "basin", "z_m"]).reset_index(drop=True)
    if not quad_df.empty:
        quad_df = quad_df.sort_values(["var", "basin", "z_m"]).reset_index(drop=True)

    lin_out  = os.path.join(out_dir, f"linear_trends_20c_21c_{mode}.csv")
    quad_out = os.path.join(out_dir, f"quad_trend_accel_fullperiod_{mode}.csv")

    lin_df.to_csv(lin_out, index=False)
    quad_df.to_csv(quad_out, index=False)

    print("[OK] Wrote:", lin_out)
    print("[OK] Wrote:", quad_out)

    if not quad_df.empty:
        print(
            "[INFO] Quadratic output summary:",
            f"rows={len(quad_df)}",
            f"basins={quad_df['basin'].nunique()}",
            f"bins={quad_df['z_m'].nunique()}",
            f"vars={sorted(quad_df['var'].unique())}",
        )

    print("[DONE]")


if __name__ == "__main__":
    main()