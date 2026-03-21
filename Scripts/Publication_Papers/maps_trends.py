#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 3: publication-style combined trend/acceleration maps
for Temperature, Salinity, and Density (PDA),
showing only:
- 0–10 m
- 1000 m

Layout
------
For each variable:
        0–10 m      1000 m
Trend     map         map
Accel     map         map

Compared to the previous version:
- only two depth columns
- larger map panels
- slightly thicker colorbars
- slightly taller figure for cleaner print layout
- keeps same I/O structure and same general aesthetics
"""

from __future__ import annotations

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon


# ==========================================================
# EMBEDDED CONFIG
# ==========================================================
IN_CSV = "/Users/elena/Projects/IRB/MedClim/Results/Trends/quad_trend_accel_fullperiod_atomic.csv"
MANCA_PY = "/Users/elena/Projects/Basins/Basins/manca.py"
OUT_DIR = "/Users/elena/Projects/IRB/MedClim/Plots/Paper_Figures"

OUT_BASENAME = "Figure3_trend_accel_surface_1000m"

# ONLY the two publication depths
DEPTH_TAGS = ["0-10", "1000"]

VARS = ["T", "S", "SIG0"]

VAR_TO_LABEL = {
    "T": "Temperature",
    "S": "Salinity",
    "SIG0": "Density (PDA)",
}
VAR_TO_PANEL = {
    "T": "a",
    "S": "b",
    "SIG0": "c",
}

DPI = 400
FIGSIZE = (14.0, 13.6)

EXTENT = (-6.5, 36.5, 30.0, 46.5)

SURFACE_BIN = (0.0, 10.0)
TOL_NEAREST_M = 2.5
CMAP = "RdBu_r"
ROBUST_Q = 98.0
ALPHA = 0.05

FIELD_TREND = "trend_mid_per_decade"
FIELD_ACCEL = "accel_per_decade2"

PVAL_CANDIDATES = {
    FIELD_TREND: [
        "trend_mid_p", "trend_p", "p_trend_mid", "p_trend",
        "pval_trend_mid", "pval_trend"
    ],
    FIELD_ACCEL: [
        "accel_p", "accel_per_decade2_p", "p_accel", "pval_accel"
    ],
}
SE_CANDIDATES = {
    FIELD_TREND: [
        "trend_mid_se_per_decade", "trend_se_per_decade",
        "trend_mid_se", "trend_se", "se_trend_mid"
    ],
    FIELD_ACCEL: [
        "accel_se_per_decade2", "accel_se", "se_accel"
    ],
}

EDGE = "k"
EDGE_LW = 0.35   # was 0.25
NO_DATA_FC = (0.70, 0.70, 0.70, 1.0)
NODATA_HATCH = "///"
NONSIG_HATCH = ".."
NONSIG_EDGE = "0.50"

# Typography
SECTION_TITLE_FS = 15.5   # variable titles (a, b, c)
COL_TITLE_FS = 13.0       # "0–10 m", "1000 m"
ROW_LABEL_FS = 12.5       # Trend / Acceleration
CBAR_LABEL_FS = 10.8
CBAR_TICK_FS = 10.0

# Figure margins
LEFT = 0.090
RIGHT = 0.915
BOTTOM = 0.038
TOP = 0.985

# Spacing between variable sections
SECTION_GAP = 0.012

# Spacing inside each variable section
TITLE_GAP = 0.009
ROW_GAP = 0.012

# Geometry of map/cbar area
INNER_GAP_X = 0.012
CBAR_W = 0.022
CBAR_PAD = 0.010

ROW_LABEL_XSHIFT = -0.060

SHOW_GLOBAL_TITLE = False
GLOBAL_TITLE = "Quadratic-fit trend and acceleration (1950–2025)"


# ==========================================================
# HELPERS
# ==========================================================
def parse_manca_py_rectangles(manca_py_path: str) -> dict[str, Polygon]:
    txt = open(manca_py_path, "r", encoding="utf-8", errors="ignore").read()
    pat = re.compile(
        r"Basin\.from_array\(\s*'(?P<code>[A-Z0-9]+)'\s*,.*?np\.array\(\s*(?P<arr>\[[\s\S]*?\])\s*\)\s*\)",
        re.DOTALL,
    )
    triple_pat = re.compile(
        r"\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*0\.?0?\s*\]"
    )
    basins = {}
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
    if (not np.isfinite(s)) or (s <= 0):
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
    d = df[(df["basin"] == basin) & (df["var"] == var)]
    if d.empty:
        return np.nan, np.nan

    if depth_tag == "0-10":
        z0, z1 = surface_bin
        ds = d[(d["z_m"] >= z0) & (d["z_m"] <= z1)]
        if ds.empty:
            return np.nan, np.nan

        val = weighted_mean(ds[field].to_numpy(), ds["n_full"].to_numpy())

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

        p_col = _detect_first_existing(df, PVAL_CANDIDATES.get(field, []))
        if p_col is not None:
            pvals = ds[p_col].to_numpy(dtype=float)
            w = ds["n_full"].to_numpy(dtype=float)
            p = weighted_mean(pvals, w)
            return val, p

        return val, np.nan

    target = float(depth_tag)
    z = d["z_m"].to_numpy(dtype=float)
    i = int(np.argmin(np.abs(z - target)))
    if np.abs(z[i] - target) > tol:
        return np.nan, np.nan

    row = d.iloc[i]
    val = float(row[field]) if field in row and np.isfinite(row[field]) else np.nan

    p_col = _detect_first_existing(df, PVAL_CANDIDATES.get(field, []))
    if p_col is not None and p_col in row:
        return val, float(row[p_col])

    se_col = _detect_first_existing(df, SE_CANDIDATES.get(field, []))
    if se_col is not None and se_col in row:
        se = float(row[se_col])
        if np.isfinite(val) and np.isfinite(se) and se > 0:
            return val, p_from_z(val / se)

    return val, np.nan


def depth_label(depth_tag: str) -> str:
    return "0–10 m" if depth_tag == "0-10" else f"{depth_tag} m"


def draw_polygon_map(ax, polygons, df, var, depth_tag, field, vlim):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#d9d6cf", edgecolor="none", zorder=5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.55, zorder=6)

    cmap = plt.get_cmap(CMAP)
    vmin, vmax = -vlim, vlim

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
            lw = EDGE_LW

            if np.isfinite(p) and (p >= ALPHA):
                hatch = NONSIG_HATCH
                edge = NONSIG_EDGE
        else:
            fc = NO_DATA_FC
            hatch = NODATA_HATCH
            edge = EDGE
            lw = 0.32

        ax.add_geometries(
            [poly],
            ccrs.PlateCarree(),
            facecolor=fc,
            edgecolor=edge,
            linewidth=lw,
            hatch=hatch,
            zorder=2
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    return sm


def collect_vlims(df: pd.DataFrame, polygons: dict[str, Polygon], var: str) -> tuple[float, float]:
    trend_vals, accel_vals = [], []
    for depth_tag in DEPTH_TAGS:
        for basin in polygons.keys():
            vt, _ = extract_value_and_p(
                df, basin, var, depth_tag, FIELD_TREND, SURFACE_BIN, TOL_NEAREST_M
            )
            va, _ = extract_value_and_p(
                df, basin, var, depth_tag, FIELD_ACCEL, SURFACE_BIN, TOL_NEAREST_M
            )
            trend_vals.append(vt)
            accel_vals.append(va)

    return (
        robust_sym_vlim(np.asarray(trend_vals), q=ROBUST_Q),
        robust_sym_vlim(np.asarray(accel_vals), q=ROBUST_Q),
    )


def add_axes_block(fig, left, right, bottom, top, ncols=2):
    """
    Create ncols map axes + 1 vertical colorbar axis
    with exactly the same height as the map row.
    """
    total_w = right - left
    map_area_w = total_w - CBAR_PAD - CBAR_W
    map_w = (map_area_w - (ncols - 1) * INNER_GAP_X) / ncols

    axes = []
    for j in range(ncols):
        x0 = left + j * (map_w + INNER_GAP_X)
        ax = fig.add_axes([x0, bottom, map_w, top - bottom], projection=ccrs.PlateCarree())
        axes.append(ax)

    cax_x = left + map_area_w + CBAR_PAD
    cax = fig.add_axes([cax_x, bottom, CBAR_W, top - bottom])
    return axes, cax


# ==========================================================
# MAIN FIGURE
# ==========================================================
def make_combined_figure(df: pd.DataFrame, polygons: dict[str, Polygon], out_base: str) -> None:
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")

    if SHOW_GLOBAL_TITLE:
        fig.suptitle(GLOBAL_TITLE, fontsize=12.5, y=0.992)

    full_h = TOP - BOTTOM
    sec_h = (full_h - 2 * SECTION_GAP) / 3.0

    for iv, var in enumerate(VARS):
        sec_top = TOP - iv * (sec_h + SECTION_GAP)
        sec_bottom = sec_top - sec_h

        title_y = sec_top - 0.008
        content_top = sec_top - TITLE_GAP
        content_bottom = sec_bottom + 0.002

        content_h = content_top - content_bottom
        row_h = (content_h - ROW_GAP) / 2.0

        trend_bottom = content_bottom + row_h + ROW_GAP
        trend_top = content_top

        accel_bottom = content_bottom
        accel_top = content_bottom + row_h

        trend_axes, cax_tr = add_axes_block(
            fig, LEFT, RIGHT, trend_bottom, trend_top, ncols=2
        )
        accel_axes, cax_ac = add_axes_block(
            fig, LEFT, RIGHT, accel_bottom, accel_top, ncols=2
        )

        vlim_tr, vlim_ac = collect_vlims(df, polygons, var)

        sm_tr = None
        sm_ac = None

        for j, depth_tag in enumerate(DEPTH_TAGS):
            sm_tr = draw_polygon_map(
                trend_axes[j], polygons, df, var, depth_tag, FIELD_TREND, vlim_tr
            )
            trend_axes[j].set_title(depth_label(depth_tag), fontsize=COL_TITLE_FS, pad=6)

            sm_ac = draw_polygon_map(
                accel_axes[j], polygons, df, var, depth_tag, FIELD_ACCEL, vlim_ac
            )

        # Section title
        fig.text(
            0.5 * (LEFT + RIGHT),
            title_y,
            f"({VAR_TO_PANEL[var]}) {VAR_TO_LABEL[var]}",
            ha="center",
            va="bottom",
            fontsize=SECTION_TITLE_FS + 0.5,
            fontweight="bold"
        )

        # Row labels
        trend_axes[0].text(
            ROW_LABEL_XSHIFT, 0.5, "Trend",
            transform=trend_axes[0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=ROW_LABEL_FS
        )
        accel_axes[0].text(
            ROW_LABEL_XSHIFT, 0.5, "Acceleration",
            transform=accel_axes[0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=ROW_LABEL_FS
        )

        # Colorbars
        cb1 = fig.colorbar(sm_tr, cax=cax_tr, orientation="vertical")
        cb1.ax.tick_params(labelsize=CBAR_TICK_FS, length=3, pad=2)
        cb1.set_label("Trend (per decade)", fontsize=CBAR_LABEL_FS, labelpad=6)

        cb2 = fig.colorbar(sm_ac, cax=cax_ac, orientation="vertical")
        cb2.ax.tick_params(labelsize=CBAR_TICK_FS, length=2, pad=2)
        cb2.set_label("Acceleration (per decade²)", fontsize=CBAR_LABEL_FS, labelpad=6)

    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    out_png = out_base + ".png"
    out_pdf = out_base + ".pdf"

    fig.savefig(out_png, dpi=DPI, facecolor="white")
    fig.savefig(out_pdf, dpi=DPI, facecolor="white")
    plt.close(fig)

    print(f"[OK] Wrote {out_png}")
    print(f"[OK] Wrote {out_pdf}")


def main():
    print("Reading:", IN_CSV)
    df = pd.read_csv(IN_CSV)
    df["basin"] = df["basin"].astype(str).str.upper().str.strip()
    df["var"] = df["var"].astype(str).str.upper().str.strip()

    print("Parsing polygons:", MANCA_PY)
    polygons = parse_manca_py_rectangles(MANCA_PY)
    print("Basins parsed:", len(polygons))

    out_base = os.path.join(OUT_DIR, OUT_BASENAME)
    make_combined_figure(df, polygons, out_base)


if __name__ == "__main__":
    main()