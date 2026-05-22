#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overall Δ(21c−20c) maps over Manca basins computed from MONTHLY MEANS,
after seasonal removal (same model as in plot_timeseries_trends.py):

Seasonal model (fixed periods 12 and 6 months):
  y_fit(t) = A*cos(2π t/12 + phi1) + B*cos(2π t/6 + phi2) + Y

Seasonally de-trended but MEAN-PRESERVING series:
  seasonal_part = y_fit - Y
  y_adj = y - seasonal_part = (y - y_fit) + Y

Then compute century means from y_adj:
  mean_20c = weighted mean of y_adj over 1950–1999 (weights = n)
  mean_21c = weighted mean of y_adj over 2000–2025 (weights = n)
  delta = mean_21c - mean_20c

Layout:
  rows = T, S, SIG0
  cols = 0–10, 200, 1000, 2000 m

No data shown as gray with hatching (distinguish from Δ≈0).
Colorbars per row: 3 groups (surface / LIW / deep).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon

from scipy.optimize import curve_fit


# ==========================================================
# HARD-CODED PATHS / SETTINGS (EDIT THESE)
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IN_CSV   = str(PROJECT_ROOT / "Results/Trends/monthly_timeseries_all_bins.csv")
MANCA_PY = str(PROJECT_ROOT.parents[1] / "Basins" / "Basins" / "manca.py")
OUTDIR   = str(PROJECT_ROOT / "Plots/MultiPlatform/DeltaMaps")

# Century windows
Y20A, Y20B = 1950, 1999
Y21A, Y21B = 2000, 2025

# Variables (rows)
VARS = ["T", "S", "SIG0"]  # SIG0 = PDA (sigma0)

# Depths (columns)
DEPTHS = [200.0, 1000.0, 2000.0]
DEPTH_TAGS = ["0-10"] + [str(int(d)) for d in DEPTHS]

# Surface range used to pool monthly 0–10 m
SURF_RANGE = (0.0, 10.0)

# For fixed depths: use nearest available z within tolerance (meters)
DEPTH_TOL = 1.0

# Coverage thresholds (per century)
MIN_N_TOTAL = 30         # total weight sum (casts) across months in each century
MIN_MONTHS  = 24         # minimum number of monthly points in each century

# Map extent
EXTENT = (-6.5, 36.5, 30.0, 46.5)

# No-data styling
NO_DATA_FC = (0.65, 0.65, 0.65, 1.0)
NO_DATA_HATCH = "///"

# Figure sizing
FIG_W_PER_COL = 3.2
FIG_H = 6.2
DPI = 240

# ==========================================================
# END SETTINGS
# ==========================================================


# --------------------------
# Geometry / Manca polygons
# --------------------------
def parse_manca_py_rectangles(manca_py_path: str) -> dict[str, Polygon]:
    """
    Robust parser for Basin.from_array(...) in manca.py.
    Handles both np.array([[...],[...]]) and np.array([ [...], ... ]) styles.
    """
    txt = open(manca_py_path, "r", encoding="utf-8", errors="ignore").read()

    pat = re.compile(
        r"Basin\.from_array\(\s*'(?P<code>[A-Z0-9]+)'\s*,.*?np\.array\(\s*(?P<arr>\[[\s\S]*?\])\s*\)\s*\)",
        re.DOTALL
    )
    triple_pat = re.compile(
        r"\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*0\.?0?\s*\]"
    )

    basins: dict[str, Polygon] = {}
    for m in pat.finditer(txt):
        code = m.group("code").strip()
        arr_text = m.group("arr")
        coords = [(float(lon), float(lat)) for lon, lat in triple_pat.findall(arr_text)]
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        basins[code] = Polygon(coords)

    if not basins:
        raise RuntimeError(f"No basins parsed from {manca_py_path}.")
    return basins


# --------------------------
# Utilities
# --------------------------
def weighted_mean(values, weights) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.sum(v[m] * w[m]) / np.sum(w[m]))


def pool_monthly(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    If there are multiple entries per month, pool them weighted by n.
    Returns: time, year, month, value, w
    """
    if df_sub.empty:
        return df_sub

    g = (
        df_sub.groupby(["time", "year", "month"], as_index=False)
              .apply(lambda x: pd.Series({
                  "value": weighted_mean(x["value"].to_numpy(), x["n"].to_numpy()),
                  "w": float(np.nansum(x["n"].to_numpy()))
              }))
              .reset_index(drop=True)
    )
    return g.sort_values("time").reset_index(drop=True)


def pick_series_for_depth(df_bv: pd.DataFrame, depth_tag: str) -> pd.DataFrame:
    """
    For a single basin+var monthly dataset (columns include time/year/month/z_m/value/n):
      - "0-10": pool all z in SURF_RANGE per month
      - numeric depth: pick nearest z within DEPTH_TOL, pool per month (if multiple entries)
    Returns dataframe with time/year/month/value/w.
    """
    if depth_tag == "0-10":
        d = df_bv[(df_bv["z_m"] >= SURF_RANGE[0]) & (df_bv["z_m"] <= SURF_RANGE[1])].copy()
        return pool_monthly(d)

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
    return pool_monthly(d)


# --------------------------
# Seasonal removal (same as trends script)
# --------------------------
def t_months_since_start(times: pd.Series) -> np.ndarray:
    """Months since first timestamp (float), consistent with 12 & 6 month periods."""
    times = pd.to_datetime(times)
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
    Fit seasonal cosine model (12 & 6 months) and return:
      y_fit_full, y_deseas_full, Y_hat
    where y_deseas = y - y_fit  (mean removed!), Y_hat is fitted mean.
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


def mean_preserving_deseason(times: pd.Series, y: np.ndarray) -> np.ndarray:
    """
    Remove only the seasonal part, keep mean:
      y_adj = y - (y_fit - Y_hat) = (y - y_fit) + Y_hat
    """
    y_fit, y_deseas, Y_hat = deseason_like_original(times, y)
    if not np.isfinite(Y_hat):
        return np.full_like(np.asarray(y, float), np.nan, dtype=float)
    return y_deseas + Y_hat


# --------------------------
# Century means & delta from monthly deseasoned series
# --------------------------
def compute_century_mean_from_series(s: pd.DataFrame, y_adj: np.ndarray, y0: int, y1: int) -> tuple[float, float, int]:
    """
    s columns: year, w ; y_adj aligned to s rows.
    Returns: (mean, n_total, n_months)
    """
    if s.empty:
        return (np.nan, 0.0, 0)

    m = (s["year"].to_numpy(int) >= y0) & (s["year"].to_numpy(int) <= y1) & np.isfinite(y_adj) & np.isfinite(s["w"].to_numpy(float))
    if not np.any(m):
        return (np.nan, 0.0, 0)

    yy = np.asarray(y_adj, float)[m]
    ww = s["w"].to_numpy(float)[m]
    mean = weighted_mean(yy, ww)
    n_total = float(np.nansum(ww))
    n_months = int(np.sum(m))
    return (mean, n_total, n_months)


def build_overall_delta_from_monthly(df_monthly: pd.DataFrame, var: str) -> pd.DataFrame:
    """
    Returns:
      basin, depth_tag, delta, mean20, mean21, n20, n21, m20, m21, status
    status in {"ok","no_data"}.
    """
    out_rows = []
    dfv = df_monthly[df_monthly["var"] == var].copy()
    if dfv.empty:
        return pd.DataFrame(columns=["basin","depth_tag","delta","mean20","mean21","n20","n21","m20","m21","status"])

    for basin, g_b in dfv.groupby("basin"):
        for depth_tag in DEPTH_TAGS:
            s = pick_series_for_depth(g_b, depth_tag)
            if s.empty:
                out_rows.append((basin, depth_tag, np.nan, np.nan, np.nan, 0.0, 0.0, 0, 0, "no_data"))
                continue

            # mean-preserving deseason
            y_adj = mean_preserving_deseason(s["time"], s["value"].to_numpy(float))

            mean20, n20, m20 = compute_century_mean_from_series(s, y_adj, Y20A, Y20B)
            mean21, n21, m21 = compute_century_mean_from_series(s, y_adj, Y21A, Y21B)

            ok = (
                np.isfinite(mean20) and np.isfinite(mean21) and
                (n20 >= MIN_N_TOTAL) and (n21 >= MIN_N_TOTAL) and
                (m20 >= MIN_MONTHS) and (m21 >= MIN_MONTHS)
            )

            delta = (mean21 - mean20) if ok else np.nan
            status = "ok" if np.isfinite(delta) else "no_data"
            out_rows.append((basin, depth_tag, delta, mean20, mean21, n20, n21, m20, m21, status))

    return pd.DataFrame(out_rows, columns=["basin","depth_tag","delta","mean20","mean21","n20","n21","m20","m21","status"])


def robust_sym_vlim(values: np.ndarray, pctl: float = 98.0, fallback: float = 1.0) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return (-fallback, fallback)
    s = np.nanpercentile(np.abs(v), pctl)
    if not np.isfinite(s) or s <= 0:
        s = fallback
    return (-float(s), float(s))


# --------------------------
# Plotting (3 rows × 5 cols), 3 colorbars per row (surface / LIW / deep)
# --------------------------
def plot_overall_3rows(delta_by_var: dict[str, pd.DataFrame],
                       basin_polys: dict[str, Polygon],
                       out_png: str):
    """
    3 rows (T,S,SIG0) × N cols (depths) with FIXED GridSpec colorbars (no drift).
    This version assumes DEPTH_TAGS = ["0-10","200","1000","2000"] (i.e., 200 m removed).

    Colorbars per row:
      - surface: column 0
      - LIW:     column 1 (200 m)
      - deep:    columns 2–3 (1000–2000 m)

    No data shown as gray + hatching.
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as mticker


    nrows = len(VARS)
    ncols = len(DEPTH_TAGS)

    fig = plt.figure(figsize=(FIG_W_PER_COL * ncols, FIG_H))

    # GridSpec: for each var row -> map row + small cbar row
    height_ratios = []
    for _ in range(nrows):
        height_ratios += [1.0, 0.08]  # map row, cbar row
    gs = gridspec.GridSpec(2 * nrows, ncols, figure=fig,
                           height_ratios=height_ratios,
                           hspace=0.05, wspace=0.02)

    axes = [[None] * ncols for _ in range(nrows)]
    for i in range(nrows):
        for j in range(ncols):
            axes[i][j] = fig.add_subplot(gs[2 * i, j], projection=ccrs.PlateCarree())

    cmap = plt.get_cmap("RdBu_r")

    # Depth-grouping now: surface (0-10), LIW (200), deep (1000,2000)
    group_cols = {
        "surface": ["0-10"],
        "liw": ["200"],
        "deep": ["1000", "2000"],
    }

    # Robust symmetric vlims per var+group
    vlims = {var: {} for var in VARS}
    for var in VARS:
        df = delta_by_var[var].copy()
        for gname, tags in group_cols.items():
            vals = df[df["depth_tag"].isin(tags)]["delta"].to_numpy(float)
            fallback = 1.0 if var == "T" else (0.2 if var == "S" else 0.1)
            vlims[var][gname] = robust_sym_vlim(vals, pctl=98.0, fallback=fallback)

    # ==========================================================
    # <<< Minor tuning knobs for cbar placement & fonts
    # ==========================================================
    CB_INSET_Y0 = 0.32   # raise bar inside its row (0..1); try 0.28–0.40
    CB_INSET_H  = 0.62   # bar height inside its row (0..1)
    CB_LABEL_FS = 7      # was 6
    CB_TICK_FS  = 9      # was 8
    TITLE_FS    = 11     # was 10
    LEFT_FS     = 12     # was 11
    # ==========================================================

    def _cbar_inset(host_ax):
        """
        host_ax is the GridSpec cell axis (we keep it invisible),
        inset is where the actual colorbar will be drawn.
        """
        host_ax.set_axis_off()
        return host_ax.inset_axes([0.04, CB_INSET_Y0, 0.92, CB_INSET_H])

    def _set_nice_ticks(cb, nticks=5):
        cb.locator = mticker.MaxNLocator(nbins=nticks)
        cb.update_ticks()

    for i, var in enumerate(VARS):
        df = delta_by_var[var].copy()
        lut = dict(zip(zip(df["basin"], df["depth_tag"]), df["delta"]))

        # ---------- MAPS (row 2*i)
        for j, dt in enumerate(DEPTH_TAGS):
            ax = axes[i][j]
            ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

            if dt == "0-10":
                vmin, vmax = vlims[var]["surface"]
            elif dt == "200":
                vmin, vmax = vlims[var]["liw"]
            else:
                vmin, vmax = vlims[var]["deep"]

            for code, poly in basin_polys.items():
                val = lut.get((code, dt), np.nan)

                if np.isfinite(val):
                    denom = (vmax - vmin) if (vmax > vmin) else 1.0
                    fc = cmap((val - vmin) / denom)
                    hatch = None
                    lw = 0.25
                else:
                    fc = NO_DATA_FC
                    hatch = NO_DATA_HATCH
                    lw = 0.30

                ax.add_geometries([poly], ccrs.PlateCarree(),
                                  facecolor=fc, edgecolor="k",
                                  linewidth=lw, hatch=hatch, zorder=2)

            ax.add_feature(cfeature.LAND, zorder=5, edgecolor="none")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=6)

            if i == 0:
                ax.set_title("Δ @ 0–10 m" if dt == "0-10" else f"Δ @ {dt} m",
                             fontsize=TITLE_FS, pad=2)  # <<<

            if j == 0:
                label = "Temperature (T)" if var == "T" else ("Salinity (S)" if var == "S" else "PDA: σ₀ (SIG0)")
                ax.text(-0.05, 0.5, label, transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=LEFT_FS)  # <<<

            ax.set_xticks([])
            ax.set_yticks([])

        # ---------- COLORBARS (row 2*i + 1) in dedicated GridSpec row
        cb_row = 2 * i + 1

        cb_ax_surf = fig.add_subplot(gs[cb_row, 0])     # surface under col 0
        cb_ax_liw  = fig.add_subplot(gs[cb_row, 1])     # 200 m under col 1
        cb_ax_deep = fig.add_subplot(gs[cb_row, 2:4])   # deep under cols 2–3

        # <<< draw the actual colorbars on raised inset axes (keeps layout identical)
        cax_surf = _cbar_inset(cb_ax_surf)
        cax_liw  = _cbar_inset(cb_ax_liw)
        cax_deep = _cbar_inset(cb_ax_deep)

        unit = "°C" if var == "T" else ("-" if var == "S" else "kg m$^{-3}$")

        labels = {
            "surface": f"Δ{var} (21c−20c) surface [{unit}]",
            "liw":     f"Δ{var} (21c−20c) LIW (~200 m) [{unit}]",
            "deep":    f"Δ{var} (21c−20c) deep (1000–2000 m) [{unit}]",
        }

        # Surface cbar
        vmin, vmax = vlims[var]["surface"]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cb = plt.colorbar(sm, cax=cax_surf, orientation="horizontal")  # <<<
        cb.ax.tick_params(labelsize=CB_TICK_FS, pad=1)                 # <<<
        cb.set_label(labels["surface"], fontsize=CB_LABEL_FS, labelpad=0)  # <<<

        # LIW cbar (200 m)
        vmin, vmax = vlims[var]["liw"]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cb = plt.colorbar(sm, cax=cax_liw, orientation="horizontal")   # <<<
        _set_nice_ticks(cb, nticks=5)   # <<< add this
        cb.ax.tick_params(labelsize=CB_TICK_FS, pad=1)                 # <<<
        cb.set_label(labels["liw"], fontsize=CB_LABEL_FS, labelpad=0)  # <<<

        # Deep cbar (1000–2000)
        vmin, vmax = vlims[var]["deep"]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cb = plt.colorbar(sm, cax=cax_deep, orientation="horizontal")  # <<<
        cb.ax.tick_params(labelsize=CB_TICK_FS, pad=1)                 # <<<
        cb.set_label(labels["deep"], fontsize=CB_LABEL_FS, labelpad=0) # <<<

    # final margins (tweak if you want)
    plt.subplots_adjust(top=0.93, bottom=0.07, left=0.06, right=0.995)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print("Saved:", out_png)


def main():
    print("Reading monthly CSV...")
    df = pd.read_csv(IN_CSV)
    df["time"] = pd.to_datetime(df["time"])

    # basic sanity
    need = {"basin","z_m","var","time","year","month","value","n"}
    miss = need.difference(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in {IN_CSV}: {sorted(miss)}")

    print("Parsing Manca polygons...")
    basin_polys = parse_manca_py_rectangles(MANCA_PY)
    print(f"Parsed {len(basin_polys)} basins.")

    delta_by_var = {}
    for var in VARS:
        print("Computing deseasoned monthly overall delta for", var)
        delta_by_var[var] = build_overall_delta_from_monthly(df, var)

    out_png = os.path.join(
        OUTDIR,
        f"DeltaMaps_OVERALL_3rows_T_S_SIG0_DESEASONEDMONTHLY_{Y21A}_{Y21B}_minus_{Y20A}_{Y20B}_0-10_200_1000_2000.png"
    )
    plot_overall_3rows(delta_by_var, basin_polys, out_png)


if __name__ == "__main__":
    main()