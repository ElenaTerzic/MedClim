#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox


# ==========================================================
# HARD-CODED INPUTS / OUTPUTS / SETTINGS
# ==========================================================

IN_CSV   = "/Users/elena/Projects/IRB/MedClim/Results/Trends/linear_trends_20c_21c_atomic.csv"
MANCA_PY = "/Users/elena/Projects/Basins/Basins/manca.py"

OUTDIR   = "/Users/elena/Projects/IRB/MedClim/Plots/MultiPlatform/Trends"
BASENAME = "LinearTrends_CoastMask"

VARS = ["T", "S", "SIG0"]
VAR_LABEL = {"T": "T", "S": "S", "SIG0": "PDA (σ₀)"}

DEPTH_TAGS = ["0-10", "200","1000", "2000"]
SURF_RANGE = (0.0, 10.0)
DEPTH_TOL  = 2.5

EXTENT = (-6.5, 36.5, 30.0, 46.5)

ROBUST_Q = 98.0
CMAP = "RdBu_r"
DPI  = 240

# No-data styling
NO_DATA_FC = (0.65, 0.65, 0.65, 1.0)
USE_HATCH_FOR_NODATA = True
NODATA_HATCH = "///"

# Insignificant styling
ALPHA = 0.05
NONSIG_HATCH = ".."
NONSIG_EDGE = "0.25"

# Layout
FIG_W = 20.5
FIG_H = 7.2
HSPACE = 0.005
WSPACE = 0.02

# Right-side colorbar
CBAR_W = 0.012
CBAR_PAD = 0.006

# Manual margins (Cartopy-friendly)
LEFT = 0.04
RIGHT = 0.92   # leave space for cbars
BOTTOM = 0.06
TOP = 0.90

# ----------------------------------------------------------
# OPTIONAL: manual limits (HIGHLY recommended for paper plots)
# If USE_MANUAL_VLIMS=True, these override the auto robust limits.
# Units are your CSV units (trend per decade).
# ----------------------------------------------------------
USE_MANUAL_VLIMS = False

MANUAL_VLIMS = {
    # ("T","surface"):(-0.8, 0.8), ("T","liw"):(-0.4, 0.4), ("T","deep"):(-0.2, 0.2),
    # ("S","surface"):(-0.10, 0.10), ("S","liw"):(-0.05, 0.05), ("S","deep"):(-0.03, 0.03),
    # ("SIG0","surface"):(-0.10, 0.10), ("SIG0","liw"):(-0.06, 0.06), ("SIG0","deep"):(-0.04, 0.04),
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


def surface_weighted_mean_and_se(vals, ses, weights):
    v = np.asarray(vals, float)
    se = np.asarray(ses, float)
    w = np.asarray(weights, float)

    m = np.isfinite(v) & np.isfinite(se) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan, np.nan

    a = w[m] / np.sum(w[m])
    mean = float(np.sum(a * v[m]))
    var = float(np.sum((a ** 2) * (se[m] ** 2)))
    return mean, (float(np.sqrt(var)) if var > 0 else np.nan)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def p_from_z(z: float) -> float:
    z = float(abs(z))
    return 2.0 * (1.0 - norm_cdf(z))


def robust_sym_vlim(values: np.ndarray, q: float, fallback: float) -> tuple[float, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-fallback, fallback)
    s = float(np.nanpercentile(np.abs(v), q))
    if not np.isfinite(s) or s <= 0:
        s = fallback
    return (-s, s)


def depth_group(tag: str) -> str:
    if tag == "0-10":
        return "surface"
    if tag in ("200"):
        return "liw"
    return "deep"


def extract_trend_value(df: pd.DataFrame, basin: str, var: str, depth_tag: str, which: str) -> float:
    d = df[(df["basin"] == basin) & (df["var"] == var)]
    if d.empty:
        return np.nan

    if depth_tag == "0-10":
        ds = d[(d["z_m"] >= SURF_RANGE[0]) & (d["z_m"] <= SURF_RANGE[1])]
        if ds.empty:
            return np.nan

        if which == "20c":
            return weighted_mean(ds["trend20_per_decade"].to_numpy(), ds["n20"].to_numpy())
        if which == "21c":
            return weighted_mean(ds["trend21_per_decade"].to_numpy(), ds["n21"].to_numpy())

        delta = ds["trend21_per_decade"].to_numpy() - ds["trend20_per_decade"].to_numpy()
        w = ds["n20"].to_numpy() + ds["n21"].to_numpy()
        return weighted_mean(delta, w)

    target = float(depth_tag)
    z = d["z_m"].to_numpy(dtype=float)
    i = int(np.argmin(np.abs(z - target)))
    if np.abs(z[i] - target) > DEPTH_TOL:
        return np.nan

    row = d.iloc[i]
    if which == "20c":
        return float(row["trend20_per_decade"])
    if which == "21c":
        return float(row["trend21_per_decade"])
    return float(row["trend21_per_decade"] - row["trend20_per_decade"])


def extract_p_value(df: pd.DataFrame, basin: str, var: str, depth_tag: str, which: str) -> float:
    if which not in ("20c", "21c"):
        return np.nan

    d = df[(df["basin"] == basin) & (df["var"] == var)]
    if d.empty:
        return np.nan

    if depth_tag == "0-10":
        ds = d[(d["z_m"] >= SURF_RANGE[0]) & (d["z_m"] <= SURF_RANGE[1])]
        if ds.empty:
            return np.nan

        if which == "20c":
            tr, se = surface_weighted_mean_and_se(
                ds["trend20_per_decade"].to_numpy(),
                ds["trend20_se_per_decade"].to_numpy(),
                ds["n20"].to_numpy(),
            )
        else:
            tr, se = surface_weighted_mean_and_se(
                ds["trend21_per_decade"].to_numpy(),
                ds["trend21_se_per_decade"].to_numpy(),
                ds["n21"].to_numpy(),
            )

        if np.isfinite(tr) and np.isfinite(se) and se > 0:
            return p_from_z(tr / se)
        return np.nan

    target = float(depth_tag)
    z = d["z_m"].to_numpy(dtype=float)
    i = int(np.argmin(np.abs(z - target)))
    if np.abs(z[i] - target) > DEPTH_TOL:
        return np.nan

    row = d.iloc[i]
    return float(row["trend20_p"]) if which == "20c" else float(row["trend21_p"])


def compute_vlims_by_var_and_group(basin_polys: dict[str, Polygon], df: pd.DataFrame, which: str):
    vals = {(var, grp): [] for var in VARS for grp in ("surface", "liw", "deep")}

    for var in VARS:
        for basin in basin_polys.keys():
            for tag in DEPTH_TAGS:
                grp = depth_group(tag)
                v = extract_trend_value(df, basin, var, tag, which)
                if np.isfinite(v):
                    vals[(var, grp)].append(v)

    vlims = {}
    for var in VARS:
        vlims[(var, "surface")] = robust_sym_vlim(np.asarray(vals[(var, "surface")]), ROBUST_Q, 0.1)
        vlims[(var, "liw")]     = robust_sym_vlim(np.asarray(vals[(var, "liw")]),     ROBUST_Q, 0.05)
        vlims[(var, "deep")]    = robust_sym_vlim(np.asarray(vals[(var, "deep")]),    ROBUST_Q, 0.02)

    # Optional manual overrides
    if USE_MANUAL_VLIMS:
        for k, lim in MANUAL_VLIMS.items():
            vlims[k] = lim

    return vlims


def combine_row_vlim(vlims, var: str) -> tuple[float, float]:
    mins = []
    maxs = []
    for grp in ("surface", "liw", "deep"):
        vmin, vmax = vlims[(var, grp)]
        mins.append(vmin)
        maxs.append(vmax)

    lo = float(min(mins))
    hi = float(max(maxs))
    m = max(abs(lo), abs(hi))
    if not np.isfinite(m) or m <= 0:
        m = 1.0
    return (-m, m)


def plot_one(which: str, title: str, out_png: str, df: pd.DataFrame, basin_polys: dict[str, Polygon]):

    vlims = compute_vlims_by_var_and_group(basin_polys, df, which)
    cmap = plt.get_cmap(CMAP)

    nrows = len(VARS)
    ncols = len(DEPTH_TAGS)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = GridSpec(nrows, ncols, figure=fig, hspace=HSPACE, wspace=WSPACE)

    row_axes: list[list[plt.Axes]] = [[] for _ in range(nrows)]

    for i, var in enumerate(VARS):
        for j, tag in enumerate(DEPTH_TAGS):
            ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
            row_axes[i].append(ax)

            ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, zorder=5, edgecolor="none")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=6)

            grp = depth_group(tag)
            vmin, vmax = vlims[(var, grp)]

            for code, poly in basin_polys.items():
                val = extract_trend_value(df, code, var, tag, which)

                if np.isfinite(val):
                    fc = cmap((val - vmin) / (vmax - vmin))
                    hatch = None
                    edge = "k"
                    lw = 0.25

                    if which in ("20c", "21c"):
                        p = extract_p_value(df, code, var, tag, which)
                        if np.isfinite(p) and (p >= ALPHA):
                            hatch = NONSIG_HATCH
                            edge = NONSIG_EDGE

                    elif which == "delta":
                        # Gate Δ display by century significance:
                        # hatch if either century is NOT significant (p >= ALPHA)
                        p20 = extract_p_value(df, code, var, tag, "20c")
                        p21 = extract_p_value(df, code, var, tag, "21c")
                        if (np.isfinite(p20) and (p20 >= ALPHA)) or (np.isfinite(p21) and (p21 >= ALPHA)):
                            hatch = NONSIG_HATCH
                            edge = NONSIG_EDGE

                else:
                    fc = NO_DATA_FC
                    hatch = NODATA_HATCH if USE_HATCH_FOR_NODATA else None
                    edge = "k"
                    lw = 0.35

                ax.add_geometries(
                    [poly], ccrs.PlateCarree(),
                    facecolor=fc, edgecolor=edge, linewidth=lw, hatch=hatch, zorder=2
                )

            if i == 0:
                ax.set_title(f"{tag} m" if tag != "0-10" else "0–10 m", fontsize=11)
            if j == 0:
                ax.text(
                    -0.06, 0.5, VAR_LABEL.get(var, var),
                    transform=ax.transAxes, rotation=90,
                    va="center", ha="right", fontsize=12, fontweight="bold"
                )

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"{title} — linear trends by basin", fontsize=15)
    plt.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP)

    # Right-side colorbar per row, aligned to row bbox
    for i, var in enumerate(VARS):
        bboxes = [a.get_position() for a in row_axes[i]]
        row_bbox = Bbox.union(bboxes)

        x0 = row_bbox.x1 + CBAR_PAD
        y0 = row_bbox.y0
        h = row_bbox.height

        ax_cb = fig.add_axes([x0, y0, CBAR_W, h])

        row_vmin, row_vmax = combine_row_vlim(vlims, var)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=row_vmin, vmax=row_vmax))
        sm.set_array([])

        cb = fig.colorbar(sm, cax=ax_cb, orientation="vertical")
        cb.ax.tick_params(labelsize=9)
        cb.set_label(f"{VAR_LABEL[var]} trend (per decade)", fontsize=9, labelpad=8)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print("Saved:", out_png)


def main():
    print("Reading:", IN_CSV)
    df = pd.read_csv(IN_CSV)

    df["basin"] = df["basin"].astype(str).str.upper().str.strip()
    df["var"]   = df["var"].astype(str).str.upper().str.strip()

    print("Parsing polygons:", MANCA_PY)
    basin_polys = parse_manca_py_rectangles(MANCA_PY)
    print("Basins parsed:", len(basin_polys))

    os.makedirs(OUTDIR, exist_ok=True)

    out20 = os.path.join(OUTDIR, f"{BASENAME}_20c.png")
    out21 = os.path.join(OUTDIR, f"{BASENAME}_21c.png")
    outd  = os.path.join(OUTDIR, f"{BASENAME}_Delta_21c_minus_20c.png")

    plot_one("20c", "20th century (1950–1999)", out20, df, basin_polys)
    plot_one("21c", "21st century (2000–2025)", out21, df, basin_polys)
    plot_one("delta", "Δ trend (21c − 20c)", outd, df, basin_polys)

    print("Done.")


if __name__ == "__main__":
    main()