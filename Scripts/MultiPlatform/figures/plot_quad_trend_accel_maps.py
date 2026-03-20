#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 2: Quadratic-fit trend + acceleration mapped by polygons.

For each depth_tag in DEPTH_TAGS, produce a 2x3 figure:
- Rows: Trend (midpoint slope per decade), Acceleration (per decade^2)
- Cols: T, S, SIG0

Input:
  quad_trend_accel_fullperiod.csv

Depth handling:
- depth_tag="0-10" -> weighted mean over z_m in [0,10], weights=n_full
- otherwise pick nearest z_m to target within tol

Styling:
- No data: gray + hatch "///"
- Non-significant: hatch ".." (p >= ALPHA), edge slightly lighter

NEW in this version:
- Per-subplot scaling: separate robust symmetric limits per (var,row)
- 6 colorbars aligned under columns (T/S/SIG0): Trend row + Accel row
"""

from __future__ import annotations

import os
import re
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon
from matplotlib.gridspec import GridSpec


# ==========================================================
# EMBEDDED CONFIG
# ==========================================================
IN_CSV  = "/Users/elena/Projects/IRB/MedClim/Results/Trends/quad_trend_accel_fullperiod_atomic.csv"
MANCA_PY = "/Users/elena/Projects/Basins/Basins/manca.py"
OUT_DIR = "/Users/elena/Projects/IRB/MedClim/Plots/MultiPlatform/Trends"

DEPTH_TAGS = ["0-10", "200", "1000", "2000"]
DPI = 220

# vars
VARS = ["T", "S", "SIG0"]

# region
EXTENT = (-6.5, 36.5, 30.0, 46.5)

# surface handling / tolerance
SURFACE_BIN = (0.0, 10.0)
TOL_NEAREST_M = 2.5

# color scaling
CMAP = "RdBu_r"
ROBUST_Q = 98.0  # symmetric |v| percentile per var+row

# significance
ALPHA = 0.05
NONSIG_HATCH = ".."
NONSIG_EDGE = "0.25"
EDGE = "k"

# no-data styling
NO_DATA_FC = (0.65, 0.65, 0.65, 1.0)
NODATA_HATCH = "///"

# layout (tighter vertical spacing)
FIGSIZE = (12, 6.2)
HSPACE = 0.06 #try 0.05-0.08 #0.015
WSPACE = 0.04

# manual margins (Cartopy-friendly)
LEFT = 0.06
RIGHT = 0.98
BOTTOM = 0.16   # leave room for 2 rows of colorbars
TOP = 0.90

# colorbar geometry under each column (figure coords; computed automatically)
CB_HEIGHT = 0.018
CB_PAD_Y = 0.010     # gap between bottom maps and top colorbar row
CB_GAP_ROWS = 0.010  # gap between trend and accel colorbar rows

# Field names for values:
FIELD_TREND = "trend_mid_per_decade"
FIELD_ACCEL = "accel_per_decade2"

# Optional: tell the script what columns store p-values / SE.
# If empty/None, it will auto-detect common names.
PVAL_CANDIDATES = {
    FIELD_TREND: [
        "trend_mid_p", "trend_p", "p_trend_mid", "p_trend", "pval_trend_mid", "pval_trend"
    ],
    FIELD_ACCEL: [
        "accel_p", "accel_per_decade2_p", "p_accel", "pval_accel"
    ],
}
SE_CANDIDATES = {
    FIELD_TREND: [
        "trend_mid_se_per_decade", "trend_se_per_decade", "trend_mid_se", "trend_se", "se_trend_mid"
    ],
    FIELD_ACCEL: [
        "accel_se_per_decade2", "accel_se", "se_accel"
    ],
}
# ==========================================================


def parse_manca_py_rectangles(manca_py_path: str) -> dict[str, Polygon]:
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
        raise RuntimeError(f"No basins parsed from {manca_py_path}")
    return basins


def weighted_mean(v, w) -> float:
    v = np.asarray(v, float)
    w = np.asarray(w, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.sum(v[m] * w[m]) / np.sum(w[m]))


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def p_from_z(z: float) -> float:
    z = float(abs(z))
    return 2.0 * (1.0 - norm_cdf(z))


def robust_sym_vlim(values: np.ndarray, q: float = 98.0) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    s = float(np.nanpercentile(np.abs(v), q))
    if not np.isfinite(s) or s <= 0:
        return 1.0
    return s


def _detect_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_value_and_p(
    df: pd.DataFrame,
    basin: str,
    var: str,
    depth_tag: str,
    field: str,
    surface_bin=(0.0, 10.0),
    tol=2.5,
) -> tuple[float, float]:
    """
    Returns (value, p) at (basin,var,depth_tag).
    p can be NaN if not available.
    """
    d = df[(df["basin"] == basin) & (df["var"] == var)]
    if d.empty:
        return np.nan, np.nan

    # pick relevant rows (surface range vs nearest)
    if depth_tag == "0-10":
        z0, z1 = surface_bin
        ds = d[(d["z_m"] >= z0) & (d["z_m"] <= z1)]
        if ds.empty:
            return np.nan, np.nan

        # surface: weighted mean for value
        val = weighted_mean(ds[field].to_numpy(), ds["n_full"].to_numpy())

        # try to compute p from SE column if present (approx z-test)
        se_col = _detect_first_existing(df, SE_CANDIDATES.get(field, []))
        if se_col is not None:
            v = ds[field].to_numpy(dtype=float)
            se = ds[se_col].to_numpy(dtype=float)
            w = ds["n_full"].to_numpy(dtype=float)

            m = np.isfinite(v) & np.isfinite(se) & np.isfinite(w) & (w > 0)
            if np.any(m):
                a = w[m] / np.sum(w[m])
                var_mean = float(np.sum((a ** 2) * (se[m] ** 2)))
                se_mean = float(np.sqrt(var_mean)) if var_mean > 0 else np.nan
                if np.isfinite(val) and np.isfinite(se_mean) and se_mean > 0:
                    return val, p_from_z(val / se_mean)

        # else try p-value column (visual gating only)
        p_col = _detect_first_existing(df, PVAL_CANDIDATES.get(field, []))
        if p_col is not None:
            pvals = ds[p_col].to_numpy(dtype=float)
            w = ds["n_full"].to_numpy(dtype=float)
            p = weighted_mean(pvals, w)
            return val, p

        return val, np.nan

    # non-surface: nearest row
    target = float(depth_tag)
    z = d["z_m"].to_numpy(dtype=float)
    i = int(np.argmin(np.abs(z - target)))
    if np.abs(z[i] - target) > tol:
        return np.nan, np.nan

    row = d.iloc[i]
    val = float(row[field]) if field in row and np.isfinite(row[field]) else np.nan

    # p-value preferred
    p_col = _detect_first_existing(df, PVAL_CANDIDATES.get(field, []))
    if p_col is not None and p_col in row:
        p = float(row[p_col])
        return val, p

    # else compute from SE if possible
    se_col = _detect_first_existing(df, SE_CANDIDATES.get(field, []))
    if se_col is not None and se_col in row:
        se = float(row[se_col])
        if np.isfinite(val) and np.isfinite(se) and se > 0:
            return val, p_from_z(val / se)

    return val, np.nan


def draw_polygon_map(
    ax,
    polygons: dict[str, Polygon],
    df: pd.DataFrame,
    var: str,
    depth_tag: str,
    field: str,
    vlim: float,
):
    """
    Per-polygon drawing (allows per-basin hatching for no-data and non-sig).
    Returns a ScalarMappable for colorbar.
    """
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, zorder=5, edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=6)

    cmap = plt.get_cmap(CMAP)
    vmin, vmax = (-vlim, vlim)

    for code, poly in polygons.items():
        val, p = extract_value_and_p(
            df, code, var, depth_tag, field,
            surface_bin=SURFACE_BIN, tol=TOL_NEAREST_M
        )

        if np.isfinite(val):
            t = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            t = float(np.clip(t, 0.0, 1.0))
            fc = cmap(t)
            hatch = None
            edge = EDGE
            lw = 0.25

            if np.isfinite(p) and (p >= ALPHA):
                hatch = NONSIG_HATCH
                edge = NONSIG_EDGE
        else:
            fc = NO_DATA_FC
            hatch = NODATA_HATCH
            edge = EDGE
            lw = 0.35

        ax.add_geometries(
            [poly], ccrs.PlateCarree(),
            facecolor=fc, edgecolor=edge, linewidth=lw, hatch=hatch, zorder=2
        )

    ax.set_xticks([])
    ax.set_yticks([])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    return sm


def depth_tag_to_fname(depth_tag: str) -> str:
    if depth_tag == "0-10":
        return "surface"
    return f"{depth_tag}m"

def _add_column_colorbars(fig, ax_top_row, ax_bottom_row, sm_tr_list, sm_ac_list):
    """
    Add 2 rows of colorbars aligned under each column:
    - Trend row: under TOP-row maps
    - Accel row: under BOTTOM-row maps

    No per-colorbar labels (keeps it clean). Adds two figure-level labels instead.
    """
    fig.canvas.draw()

    # We'll keep track of the y positions to place figure-level labels nicely
    y_tr_centers = []
    y_ac_centers = []

    for j in range(3):
        pos_top = ax_top_row[j].get_position()
        pos_bot = ax_bottom_row[j].get_position()

        x0 = pos_bot.x0
        w  = pos_bot.width

        y_tr = pos_top.y0 - CB_PAD_Y - CB_HEIGHT
        y_ac = pos_bot.y0 - CB_PAD_Y - CB_HEIGHT

        # Trend cbar
        cax_tr = fig.add_axes([x0, y_tr, w, CB_HEIGHT])
        cb_tr = fig.colorbar(sm_tr_list[j], cax=cax_tr, orientation="horizontal")
        cb_tr.ax.tick_params(labelsize=8)

        # Accel cbar
        cax_ac = fig.add_axes([x0, y_ac, w, CB_HEIGHT])
        cb_ac = fig.colorbar(sm_ac_list[j], cax=cax_ac, orientation="horizontal")
        cb_ac.ax.tick_params(labelsize=8)

        y_tr_centers.append(y_tr + 0.5 * CB_HEIGHT)
        y_ac_centers.append(y_ac + 0.5 * CB_HEIGHT)

    # -------- Figure-level labels centered under the entire row of CBs
    # Use center of middle column for x, and median y for stability
    x_center = ax_bottom_row[1].get_position().x0 + 0.5 * ax_bottom_row[1].get_position().width
    y_tr_label = float(np.median(y_tr_centers)) - 0.05
    y_ac_label = float(np.median(y_ac_centers)) - 0.05

    fig.text(x_center, y_tr_label, "Trend (per decade)", ha="center", va="top", fontsize=10)
    fig.text(x_center, y_ac_label, "Acceleration (per decade²)", ha="center", va="top", fontsize=10)


def make_one_figure(df: pd.DataFrame, polygons: dict[str, Polygon], depth_tag: str, out_png: str):
    # ======================================================
    # Robust limits per variable (separate for trend vs accel)
    # ======================================================
    vlim_tr: dict[str, float] = {}
    vlim_ac: dict[str, float] = {}

    for var in VARS:
        trend_vals = []
        accel_vals = []
        for basin in polygons.keys():
            v_tr, _ = extract_value_and_p(
                df, basin, var, depth_tag, FIELD_TREND,
                surface_bin=SURFACE_BIN, tol=TOL_NEAREST_M
            )
            v_ac, _ = extract_value_and_p(
                df, basin, var, depth_tag, FIELD_ACCEL,
                surface_bin=SURFACE_BIN, tol=TOL_NEAREST_M
            )
            trend_vals.append(v_tr)
            accel_vals.append(v_ac)

        vlim_tr[var] = robust_sym_vlim(np.asarray(trend_vals), q=ROBUST_Q)
        vlim_ac[var] = robust_sym_vlim(np.asarray(accel_vals), q=ROBUST_Q)

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(nrows=2, ncols=3, figure=fig, wspace=WSPACE, hspace=HSPACE)

    # store one ScalarMappable per column for each row (trend/accel)
    sm_tr_list = [None, None, None]
    sm_ac_list = [None, None, None]
    ax_top_row = [None, None, None]
    ax_bottom_row = [None, None, None]

    for j, var in enumerate(VARS):
        # Row 0: trend
        ax0 = fig.add_subplot(gs[0, j], projection=ccrs.PlateCarree())
        sm0 = draw_polygon_map(
            ax0, polygons, df, var, depth_tag, FIELD_TREND, vlim=vlim_tr[var]
        )
        sm_tr_list[j] = sm0
        ax0.set_title(f"{var}", fontsize=11)

        if j == 0:
            ax0.text(-0.08, 0.5, "Trend (per decade)", transform=ax0.transAxes,
                     rotation=90, va="center", ha="right", fontsize=10)

        # Row 1: acceleration
        ax1 = fig.add_subplot(gs[1, j], projection=ccrs.PlateCarree())
        sm1 = draw_polygon_map(
            ax1, polygons, df, var, depth_tag, FIELD_ACCEL, vlim=vlim_ac[var]
        )
        sm_ac_list[j] = sm1
        ax_top_row[j] = ax0
        ax_bottom_row[j] = ax1

        if j == 0:
            ax1.text(-0.08, 0.5, "Acceleration (per decade²)", transform=ax1.transAxes,
                     rotation=90, va="center", ha="right", fontsize=10)

    fig.suptitle(
        f"Quadratic fit (1950–2025): trend & acceleration — depth {depth_tag} m",
        fontsize=13, y=0.98
    )
    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP)

    # ======================================================
    # 6 colorbars aligned under columns (T/S/SIG0)
    # ======================================================
    _add_column_colorbars(fig, ax_top_row, ax_bottom_row, sm_tr_list, sm_ac_list)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=DPI)#, bbox_inches="tight"
    plt.close(fig)
    print(f"[OK] Wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--depth_tags", nargs="*", default=None,
                    help="Optional override, e.g. --depth_tags 0-10 200 1000 2000")
    args = ap.parse_args()

    depth_tags = args.depth_tags if args.depth_tags else DEPTH_TAGS

    print("Reading:", IN_CSV)
    df = pd.read_csv(IN_CSV)
    df["basin"] = df["basin"].astype(str).str.upper().str.strip()
    df["var"] = df["var"].astype(str).str.upper().str.strip()

    print("Parsing polygons:", MANCA_PY)
    polygons = parse_manca_py_rectangles(MANCA_PY)
    print("Basins parsed:", len(polygons))

    os.makedirs(OUT_DIR, exist_ok=True)

    for depth_tag in depth_tags:
        out_name = f"Fig2_quad_trend_accel_{depth_tag_to_fname(depth_tag)}.png"
        out_png = os.path.join(OUT_DIR, out_name)
        make_one_figure(df, polygons, depth_tag, out_png)


if __name__ == "__main__":
    main()