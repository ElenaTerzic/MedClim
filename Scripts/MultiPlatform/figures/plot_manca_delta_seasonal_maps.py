#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reads : Stats/MultiSource_Climatology_BINNED_ELIG/*/POOLED_RAW/climatology_POOLED_RAW_*_atomic_after_sigma.csv; Basins/manca.py
# Writes: Plots/MultiPlatform/DeltaMaps/DeltaMaps_polygons_<var>_21c-20c_0-10_200_1000_2000.png

"""
Hardcoded version (no CLI args):
Δ(21c−20c) Manca polygon maps (no grid), per season and depth.

Inputs: pooled climatology CSVs (20c, 21c) and manca.py with Basin.from_array polygons.
Outputs: PNGs for T, S, SIG0.

Rows = seasons (JFM, AMJ, JAS, OND)
Cols = depths: 0–10, 200, 1000, 2000

Colorbars (3):
  - surface (0–10)
  - LIW (200)
  - deep (1000–2000)
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from shapely.geometry import Polygon
from matplotlib.gridspec import GridSpec

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Scripts.paths import STATS, PLOTS, MANCA_PY


# ==========================================================
# HARD-CODED INPUTS / OUTPUTS / SETTINGS
# ==========================================================

CLIM20_CSV = str(STATS / "MultiSource_Climatology_BINNED_ELIG" / "20c_1950_1999" / "POOLED_RAW" / "climatology_POOLED_RAW_20c_1950_1999_atomic_after_sigma.csv")
CLIM21_CSV = str(STATS / "MultiSource_Climatology_BINNED_ELIG" / "21c_2000_2025" / "POOLED_RAW" / "climatology_POOLED_RAW_21c_2000_2025_atomic_after_sigma.csv")


OUTDIR     = str(PLOTS / "MultiPlatform" / "DeltaMaps")

# Variables to plot (one PNG per var)
VARS = ["T", "S", "SIG0"]

# Depth columns (meters)
DEPTHS = [200.0, 1000.0, 2000.0]
DEPTH_TAGS = ["0-10"] + [str(int(d)) for d in DEPTHS]

# Surface bin definition (use your existing bin centers within this window, usually 5 & 10 m)
SURFACE_BIN = (0.0, 10.0)

# Masks / QC
MIN_N = 5                 # minimum n in BOTH centuries
MASK_LOWCOV_FLAGS = True  # use flag_T_lowcov / flag_S_lowcov if present

# Map extent: (lonmin, lonmax, latmin, latmax)
EXTENT = (-6.5, 36.5, 30.0, 46.5)

# Output image settings
DPI = 220

# ==========================================================
# END SETTINGS
# ==========================================================


SEASON_ORDER = ["JFM", "AMJ", "JAS", "OND"]
SEASON_LABEL = {"JFM": "Winter (JFM)", "AMJ": "Spring (AMJ)", "JAS": "Summer (JAS)", "OND": "Autumn (OND)"}


def parse_manca_py_rectangles(manca_py_path: str) -> dict[str, Polygon]:
    """
    Robust parser for Basin.from_array(...) in manca.py.
    Handles both:
      np.array([[...],[...]])
    and
      np.array([
          [...],
          ...
      ])
    and ignores inline comments.
    """
    txt = open(manca_py_path, "r", encoding="utf-8", errors="ignore").read()

    # capture CODE + full np.array([ ... ]) block (single-bracket form too)
    pat = re.compile(
        r"Basin\.from_array\(\s*'(?P<code>[A-Z0-9]+)'\s*,.*?np\.array\(\s*(?P<arr>\[[\s\S]*?\])\s*\)\s*\)",
        re.DOTALL
    )

    # extract all [lon, lat, 0] triples inside that array block
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
        raise RuntimeError(f"No basins parsed from {manca_py_path}. Check file / regex.")

    return basins


def weighted_mean(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.sum(v[m] * w[m]) / np.sum(w[m]))


def build_delta_table(df20: pd.DataFrame, df21: pd.DataFrame, var: str,
                      depths: list[float],
                      surface_bin: tuple[float, float],
                      min_n: int,
                      mask_lowcov: bool) -> pd.DataFrame:
    """
    Returns rows: basin, season, depth_tag, delta
    """
    assert var in ("T", "S", "SIG0")

    mean_col = f"{var}_mean"
    n_col = f"n_{var}"
    flag_col = f"flag_{var}_lowcov" if var in ("T", "S") else None

    keep = ["basin", "season", "z_m", mean_col, n_col]
    if flag_col and flag_col in df20.columns and flag_col in df21.columns:
        keep += [flag_col]

    d20 = df20[keep].copy()
    d21 = df21[keep].copy()

    def extract_at_depth(d, depth_m):
        sub = d.loc[d["z_m"] == depth_m, ["basin", "season", mean_col, n_col] + ([flag_col] if flag_col and flag_col in d.columns else [])].copy()
        sub = sub.rename(columns={mean_col: "mean", n_col: "n"})
        if flag_col and flag_col in sub.columns:
            sub = sub.rename(columns={flag_col: "flag_lowcov"})
        else:
            sub["flag_lowcov"] = 0
        return sub

    def apply_masks(m):
        ok = (m["n20"] >= min_n) & (m["n21"] >= min_n)
        if mask_lowcov:
            ok &= (m["flag_lowcov20"] == 0) & (m["flag_lowcov21"] == 0)
        m.loc[~ok, "delta"] = np.nan
        return m

    out = []

    # --- surface 0–10: weighted mean across all bins within window (usually z_m=5 and 10)
    z0, z1 = surface_bin
    surf20 = d20[(d20["z_m"] >= z0) & (d20["z_m"] <= z1)].copy()
    surf21 = d21[(d21["z_m"] >= z0) & (d21["z_m"] <= z1)].copy()

    def make_surface(dd):
        rows = []
        for (basin, season), g in dd.groupby(["basin", "season"]):
            val = weighted_mean(g[mean_col], g[n_col])
            nsum = int(np.nansum(g[n_col].values))
            flg = int(np.nanmax(g[flag_col].values)) if (flag_col and flag_col in g.columns) else 0
            rows.append((basin, season, val, nsum, flg))
        return pd.DataFrame(rows, columns=["basin", "season", "mean", "n", "flag_lowcov"])

    s20 = make_surface(surf20)
    s21 = make_surface(surf21)

    m = s20.merge(s21, on=["basin", "season"], suffixes=("20", "21"), how="outer")
    m["delta"] = m["mean21"] - m["mean20"]
    m = apply_masks(m)
    for _, r in m.iterrows():
        out.append((r["basin"], r["season"], "0-10", r["delta"]))

    # --- fixed depths
    for depth_m in depths:
        a20 = extract_at_depth(d20, depth_m)
        a21 = extract_at_depth(d21, depth_m)
        m = a20.merge(a21, on=["basin", "season"], suffixes=("20", "21"), how="outer")
        m["delta"] = m["mean21"] - m["mean20"]
        m = apply_masks(m)
        tag = str(int(depth_m))
        for _, r in m.iterrows():
            out.append((r["basin"], r["season"], tag, r["delta"]))

    return pd.DataFrame(out, columns=["basin", "season", "depth_tag", "delta"])


def robust_sym_vlim(values: np.ndarray, pctl: float = 98.0, fallback: float = 1.0) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return (-fallback, fallback)
    s = np.nanpercentile(np.abs(v), pctl)
    if not np.isfinite(s) or s <= 0:
        s = fallback
    return (-float(s), float(s))


def plot_delta_polygons(delta_df: pd.DataFrame,
                       basin_polys: dict[str, Polygon],
                       var: str,
                       depth_tags: list[str],
                       out_png: str,
                       extent,
                       dpi=220):
    """
    4 rows (seasons) × N columns (depth tags)
    with 3 colorbars: surface, LIW(200/300), deep(1000/2000)
    """
    lut = {}
    for (season, depth_tag), g in delta_df.groupby(["season", "depth_tag"]):
        lut[(season, depth_tag)] = dict(zip(g["basin"], g["delta"]))

    vals_surface = delta_df.loc[delta_df["depth_tag"] == "0-10", "delta"].values.astype(float)
    vals_liw = delta_df.loc[delta_df["depth_tag"].isin(["200"]), "delta"].values.astype(float)
    vals_deep = delta_df.loc[delta_df["depth_tag"].isin(["1000", "2000"]), "delta"].values.astype(float)

    vlim_surface = robust_sym_vlim(vals_surface)
    vlim_liw = robust_sym_vlim(vals_liw)
    vlim_deep = robust_sym_vlim(vals_deep)

    def vlim_for_col(dt: str):
        if dt == "0-10":
            return vlim_surface
        if dt in ("200"):
            return vlim_liw
        return vlim_deep

    cmap = plt.get_cmap("RdBu_r")
    nrows, ncols = 4, len(depth_tags)

    fig = plt.figure(figsize=(3.2 * ncols, 2.0 * nrows))
    gs = GridSpec(nrows + 1, ncols, figure=fig,
                  height_ratios=[1, 1, 1, 1, 0.10],
                  hspace=0.005, wspace=0.02)

    for i, season in enumerate(SEASON_ORDER):
        for j, dt in enumerate(depth_tags):
            ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())

            ax.add_feature(cfeature.LAND, zorder=5, edgecolor="none")     
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=6)   

            vmin, vmax = vlim_for_col(dt)
            basin2delta = lut.get((season, dt), {})

            for code, poly in basin_polys.items():
                val = basin2delta.get(code, np.nan)
                NO_DATA_FC = (0.65, 0.65, 0.65, 1.0)  # medium gray (clearly not "zero")
                ZERO_FC    = None  # keep cmap for zero

                if np.isfinite(val):
                    fc = cmap((val - vmin) / (vmax - vmin))
                    ec = "k"
                    lw = 0.25
                    hatch = None
                else:
                    fc = NO_DATA_FC
                    ec = "k"
                    lw = 0.35
                    hatch = "///"   # optional: comment out if you want only gray

                ax.add_geometries([poly], ccrs.PlateCarree(),
                                  facecolor=fc, edgecolor="k", hatch=hatch, linewidth=0.25, zorder=2)

            if i == 0:
                title = f"{var} Δ @ 0–10 m" if dt == "0-10" else f"{var} Δ @ {dt} m"
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.text(-0.02, 0.5, SEASON_LABEL[season], transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

    # --- 3 colorbars in bottom row
    # surface: col 0
    ax_cb_surf = fig.add_subplot(gs[nrows, 0])
    ax_cb_surf.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vlim_surface[0], vmax=vlim_surface[1]))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_cb_surf, orientation="horizontal", fraction=1.0, pad=0.0)
    cb.ax.tick_params(labelsize=8)
    cb.set_label(f"Δ{var} (21c−20c) surface", fontsize=9)

    # LIW: columns for 200–300 if both exist
    if "200" in depth_tags:
        j1 = depth_tags.index("200")
        ax_cb_liw = fig.add_subplot(gs[nrows, j1])
        ax_cb_liw.axis("off")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vlim_liw[0], vmax=vlim_liw[1]))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax_cb_liw, orientation="horizontal", fraction=1.0, pad=0.0)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(f"Δ{var} (21c−20c) LIW (200 m)", fontsize=9)

    # deep: columns for 1000–2000 if both exist
    if "1000" in depth_tags and "2000" in depth_tags:
        j1 = depth_tags.index("1000")
        j2 = depth_tags.index("2000")
        ax_cb_deep = fig.add_subplot(gs[nrows, j1:j2 + 1])
        ax_cb_deep.axis("off")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vlim_deep[0], vmax=vlim_deep[1]))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax_cb_deep, orientation="horizontal", fraction=1.0, pad=0.0)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(f"Δ{var} (21c−20c) deep (1000–2000 m)", fontsize=9)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.subplots_adjust(top=0.98, bottom=0.08, left=0.05, right=0.995, hspace=0.02, wspace=0.02)
    plt.savefig(out_png, dpi=dpi) #, bbox_inches="tight"
    plt.close(fig)
    print("Saved:", out_png)


def run_one(var: str, df20: pd.DataFrame, df21: pd.DataFrame, basin_polys: dict[str, Polygon]):
    delta = build_delta_table(
        df20, df21, var=var,
        depths=DEPTHS,
        surface_bin=SURFACE_BIN,
        min_n=MIN_N,
        mask_lowcov=MASK_LOWCOV_FLAGS
    )

    out_png = os.path.join(OUTDIR, f"DeltaMaps_polygons_{var}_21c-20c_0-10_200_1000_2000.png")
    plot_delta_polygons(
        delta_df=delta,
        basin_polys=basin_polys,
        var=var,
        depth_tags=DEPTH_TAGS,
        out_png=out_png,
        extent=EXTENT,
        dpi=DPI
    )


def main():
    print("Reading climatology CSVs...")
    df20 = pd.read_csv(CLIM20_CSV)
    df21 = pd.read_csv(CLIM21_CSV)

    print("Parsing Manca polygons from manca.py...")
    basin_polys = parse_manca_py_rectangles(MANCA_PY)
    print(f"Parsed {len(basin_polys)} basins from manca.py")

    for var in VARS:
        print(f"Plotting {var} ...")
        run_one(var, df20, df21, basin_polys)

    print("Done.")


if __name__ == "__main__":
    main()