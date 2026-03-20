#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MapPlotter satellite maps of Manca Mediterranean basins,
with coast-following masking of LAND areas.

Outputs
-------
1) atomic_manca_basins_satellite_coastmasked.png
2) composed_manca_basins_satellite_coastmasked.png

Method
------
- Basin membership is computed from Basins.manca.* objects using basin.areinside(xyz)
- A land mask is built from Natural Earth land polygons (via cartopy)
- Final plotted mask is:
      final_mask = basin_mask & (~land_mask)

This keeps the Manca offshore boundaries, but clips the basin fills/outlines
to the sea along real coastlines.
"""

from __future__ import annotations

from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as MplPath

import MapPlotter as mp
import Basins

import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep


# ==========================================================
# OUTPUT / PATHS
# ==========================================================

OUT_DIR = Path("/Users/elena/Projects/IRB/MedClim/Plots/BasinMaps_MapPlotter")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ATOMIC = OUT_DIR / "atomic_manca_basins_satellite_coastmasked.png"
OUT_COMPOSED = OUT_DIR / "composed_manca_basins_satellite_coastmasked.png"


# ==========================================================
# MAP SETTINGS
# ==========================================================

MEDITERRANEAN_XLIM = [-7.2, 37.8]
MEDITERRANEAN_YLIM = [29.0, 46.5]

MAP_ZOOM = 5
FIGSIZE = (15.5, 9.2)
DPI = 180

# Basin reconstruction grid
DLON = 0.04
DLAT = 0.04

# Label sizes
ATOMIC_LABEL_SIZE = 8.5
COMPOSED_LABEL_SIZE = 10.5

# Styling
ATOMIC_ALPHA = 0.32
COMPOSED_ALPHA = 0.34

ATOMIC_OUTLINE_COLOR = "white"
COMPOSED_OUTLINE_COLOR = "white"

ATOMIC_OUTLINE_LW = 0.9
COMPOSED_OUTLINE_LW = 1.4

SHOW_MEMBER_OUTLINES_ON_COMPOSED = True
MEMBER_OUTLINE_LW = 0.7
MEMBER_OUTLINE_ALPHA = 0.40

# Optional: slightly smooth very pixelated coastal edges
# 0 = no smoothing, 1 = tiny cleanup
SMOOTHING_PASSES = 1


# ==========================================================
# ATOMIC BASINS
# ==========================================================

ATOMIC_BASINS = OrderedDict([
    ("DF2", Basins.manca.df2),
    ("DF3", Basins.manca.df3),
    ("DF4", Basins.manca.df4),
    ("DS2", Basins.manca.ds2),
    ("DF1", Basins.manca.df1),
    ("DS1", Basins.manca.ds1),
    ("DS3", Basins.manca.ds3),
    ("DS4", Basins.manca.ds4),
    ("DT1", Basins.manca.dt1),
    ("DT2", Basins.manca.dt2),
    ("DT3", Basins.manca.dt3),
    ("DI1", Basins.manca.di1),
    ("DI3", Basins.manca.di3),
    ("DJ1", Basins.manca.dj1),
    ("DJ2", Basins.manca.dj2),
    ("DJ3", Basins.manca.dj3),
    ("DJ6", Basins.manca.dj6),
    ("DJ4", Basins.manca.dj4),
    ("DJ7", Basins.manca.dj7),
    ("DJ8", Basins.manca.dj8),
    ("DJ5", Basins.manca.dj5),
    ("DH1", Basins.manca.dh1),
    ("DH2", Basins.manca.dh2),
    ("DH3", Basins.manca.dh3),
    ("DL1", Basins.manca.dl1),
    ("DL2", Basins.manca.dl2),
    ("DL3", Basins.manca.dl3),
    ("DL4", Basins.manca.dl4),
])

COMPOSED_BASINS = OrderedDict([
    ("alb",  {"name": "Alboran",           "members": ["DS1"]}),
    ("west", {"name": "Western Med",       "members": ["DS2", "DS3", "DS4", "DI1"]}),
    ("nwm",  {"name": "North-Western Med", "members": ["DF1", "DF2", "DF3", "DF4"]}),
    ("tyr",  {"name": "Tyrrhenian",        "members": ["DT1", "DT2", "DT3"]}),
    ("sic",  {"name": "Sicily Strait",     "members": ["DI3"]}),
    ("nadr", {"name": "Northern Adriatic", "members": ["DJ1"]}),
    ("madr", {"name": "Mid Adriatic",      "members": ["DJ2"]}),
    ("sadr", {"name": "Southern Adriatic", "members": ["DJ3"]}),
    ("ion",  {"name": "Ionian",            "members": ["DJ4", "DJ5", "DJ6", "DJ7", "DJ8"]}),
    ("lev",  {"name": "Levantine",         "members": ["DL1", "DL2", "DL3", "DL4", "DH3"]}),
    ("naeg", {"name": "Northern Aegean",   "members": ["DH1"]}),
    ("saeg", {"name": "Southern Aegean",   "members": ["DH2"]}),
])


# ==========================================================
# COLORS
# ==========================================================

ATOMIC_COLORS = OrderedDict([
    ("DF2", "#7fc97f"),
    ("DF3", "#4daf4a"),
    ("DF4", "#2b8cbe"),
    ("DS2", "#fdae61"),
    ("DF1", "#66c2a5"),
    ("DS1", "#d7191c"),
    ("DS3", "#f46d43"),
    ("DS4", "#fdae61"),
    ("DT1", "#74add1"),
    ("DT2", "#4575b4"),
    ("DT3", "#5e4fa2"),
    ("DI1", "#fee08b"),
    ("DI3", "#fdb863"),
    ("DJ1", "#abd9e9"),
    ("DJ2", "#2c7bb6"),
    ("DJ3", "#313695"),
    ("DJ6", "#78c679"),
    ("DJ4", "#41ab5d"),
    ("DJ7", "#238443"),
    ("DJ8", "#006837"),
    ("DJ5", "#004529"),
    ("DH1", "#f781bf"),
    ("DH2", "#c51b8a"),
    ("DH3", "#8e0152"),
    ("DL1", "#80b1d3"),
    ("DL2", "#377eb8"),
    ("DL3", "#6baed6"),
    ("DL4", "#08519c"),
])

COMPOSED_COLORS = OrderedDict([
    ("alb",  "#d7191c"),
    ("west", "#f46d43"),
    ("nwm",  "#1a9641"),
    ("tyr",  "#2b83ba"),
    ("sic",  "#fdae61"),
    ("nadr", "#abd9e9"),
    ("madr", "#2c7bb6"),
    ("sadr", "#313695"),
    ("ion",  "#238443"),
    ("lev",  "#2166ac"),
    ("naeg", "#dd1c77"),
    ("saeg", "#980043"),
])


# ==========================================================
# HELPERS
# ==========================================================

def hex_to_rgba01(hex_color: str, alpha: float):
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16) / 255.0,
        int(hex_color[2:4], 16) / 255.0,
        int(hex_color[4:6], 16) / 255.0,
        alpha,
    )


def build_mapplotter_axes(xlim, ylim, figsize=FIGSIZE, map_zoom=MAP_ZOOM):
    plotter = mp.MapPlotter(projection="PlateCarree")

    params = plotter.defaultParams()
    params["dpi"] = DPI
    params["xlim"] = list(xlim)
    params["ylim"] = list(ylim)
    params["features"] = ["continents", "tilemap"]
    params["map_zoom"] = map_zoom
    params["map_style"] = "satellite"
    params["xlabel"] = ["Longitude (deg)"]
    params["ylabel"] = ["Latitude (deg)"]
    params["draw_cbar"] = False

    plotter.plot_empty(params=params, clear=True)

    fig = plotter._fig
    ax = plotter._ax

    try:
        fig.set_size_inches(figsize[0], figsize[1], forward=True)
    except Exception:
        pass

    return plotter, fig, ax


def build_grid(xlim, ylim, dlon=DLON, dlat=DLAT):
    lon = np.arange(xlim[0], xlim[1] + dlon, dlon)
    lat = np.arange(ylim[0], ylim[1] + dlat, dlat)
    lon2, lat2 = np.meshgrid(lon, lat)

    xyz = np.zeros((lon2.size, 3), dtype=float)
    xyz[:, 0] = lon2.ravel()
    xyz[:, 1] = lat2.ravel()

    return lon, lat, lon2, lat2, xyz


def basin_mask_on_grid(basin, xyz, ny, nx):
    mask = basin.areinside(xyz)
    return np.asarray(mask, dtype=bool).reshape(ny, nx)


def label_xy_from_mask(mask, lon2, lat2):
    iy, ix = np.where(mask)
    if len(ix) == 0:
        return None, None
    return float(np.median(lon2[iy, ix])), float(np.median(lat2[iy, ix]))


def draw_mask_outline(ax, lon, lat, mask, color="white", lw=1.0, alpha=1.0, zorder=30):
    zz = np.where(mask, 1.0, np.nan)
    ax.contour(
        lon, lat, zz,
        levels=[0.5],
        colors=[color],
        linewidths=[lw],
        alpha=alpha,
        zorder=zorder,
    )


def add_label(ax, x, y, text, fontsize):
    if x is None or y is None:
        return
    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="white",
        zorder=50,
        path_effects=[
            pe.Stroke(linewidth=2.2, foreground="black"),
            pe.Normal(),
        ],
    )


# ==========================================================
# LAND MASK
# ==========================================================

def build_land_mask(lon2, lat2):
    """
    Build a boolean land mask on the plotting grid using Natural Earth land polygons.
    True  = land
    False = not land
    """
    shpfilename = shpreader.natural_earth(
        resolution="10m",
        category="physical",
        name="land",
    )

    reader = shpreader.Reader(shpfilename)
    geoms = list(reader.geometries())

    # Merge and prepare for faster point-in-polygon tests
    land_union = unary_union(geoms)
    land_prepared = prep(land_union)

    pts = np.column_stack([lon2.ravel(), lat2.ravel()])

    mask_flat = np.zeros(pts.shape[0], dtype=bool)
    for i, (x, y) in enumerate(pts):
        mask_flat[i] = land_prepared.contains(Point(x, y))

    return mask_flat.reshape(lon2.shape)


def smooth_mask(mask, passes=1):
    """
    Very small cleanup of isolated single-grid cells.
    No scipy dependency.
    """
    if passes <= 0:
        return mask.copy()

    out = mask.copy()
    for _ in range(passes):
        padded = np.pad(out.astype(int), 1, mode="edge")
        neigh = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        )
        # keep cells with >= 4 occupied neighbors, add cells with >= 5
        out = neigh >= 5
    return out


# ==========================================================
# PLOTTING
# ==========================================================

def plot_atomic():
    _, fig, ax = build_mapplotter_axes(
        xlim=MEDITERRANEAN_XLIM,
        ylim=MEDITERRANEAN_YLIM,
        figsize=FIGSIZE,
        map_zoom=MAP_ZOOM,
    )

    lon, lat, lon2, lat2, xyz = build_grid(MEDITERRANEAN_XLIM, MEDITERRANEAN_YLIM)
    ny, nx = lon2.shape

    print("Building land mask...")
    land_mask = build_land_mask(lon2, lat2)
    sea_mask = ~land_mask

    for code, basin in ATOMIC_BASINS.items():
        basin_mask = basin_mask_on_grid(basin, xyz, ny, nx)
        final_mask = basin_mask & sea_mask

        if SMOOTHING_PASSES > 0:
            final_mask = smooth_mask(final_mask, passes=SMOOTHING_PASSES)

        rgba = hex_to_rgba01(ATOMIC_COLORS[code], ATOMIC_ALPHA)
        cmap = ListedColormap([(0, 0, 0, 0), rgba])

        ax.imshow(
            final_mask.astype(int),
            origin="lower",
            extent=[lon.min(), lon.max(), lat.min(), lat.max()],
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            zorder=18,
            aspect="auto",
        )

        draw_mask_outline(
            ax, lon, lat, final_mask,
            color=ATOMIC_OUTLINE_COLOR,
            lw=ATOMIC_OUTLINE_LW,
            alpha=0.95,
            zorder=28,
        )

        xlab, ylab = label_xy_from_mask(final_mask, lon2, lat2)
        add_label(ax, xlab, ylab, code, fontsize=ATOMIC_LABEL_SIZE)

    ax.set_title("Mediterranean atomic Manca subbasins", fontsize=15, fontweight="bold", pad=10)

    fig.savefig(OUT_ATOMIC, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_ATOMIC}")


def plot_composed():
    _, fig, ax = build_mapplotter_axes(
        xlim=MEDITERRANEAN_XLIM,
        ylim=MEDITERRANEAN_YLIM,
        figsize=FIGSIZE,
        map_zoom=MAP_ZOOM,
    )

    lon, lat, lon2, lat2, xyz = build_grid(MEDITERRANEAN_XLIM, MEDITERRANEAN_YLIM)
    ny, nx = lon2.shape

    print("Building land mask...")
    land_mask = build_land_mask(lon2, lat2)
    sea_mask = ~land_mask

    atomic_masks = {}
    for code, basin in ATOMIC_BASINS.items():
        basin_mask = basin_mask_on_grid(basin, xyz, ny, nx)
        final_mask = basin_mask & sea_mask
        if SMOOTHING_PASSES > 0:
            final_mask = smooth_mask(final_mask, passes=SMOOTHING_PASSES)
        atomic_masks[code] = final_mask

    if SHOW_MEMBER_OUTLINES_ON_COMPOSED:
        for code, mask in atomic_masks.items():
            draw_mask_outline(
                ax, lon, lat, mask,
                color="white",
                lw=MEMBER_OUTLINE_LW,
                alpha=MEMBER_OUTLINE_ALPHA,
                zorder=22,
            )

    for key, info in COMPOSED_BASINS.items():
        members = info["members"]
        comp_mask = np.zeros((ny, nx), dtype=bool)
        for m in members:
            comp_mask |= atomic_masks[m]

        rgba = hex_to_rgba01(COMPOSED_COLORS[key], COMPOSED_ALPHA)
        cmap = ListedColormap([(0, 0, 0, 0), rgba])

        ax.imshow(
            comp_mask.astype(int),
            origin="lower",
            extent=[lon.min(), lon.max(), lat.min(), lat.max()],
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            zorder=20,
            aspect="auto",
        )

        draw_mask_outline(
            ax, lon, lat, comp_mask,
            color=COMPOSED_OUTLINE_COLOR,
            lw=COMPOSED_OUTLINE_LW,
            alpha=0.98,
            zorder=30,
        )

        xlab, ylab = label_xy_from_mask(comp_mask, lon2, lat2)
        add_label(ax, xlab, ylab, info["name"], fontsize=COMPOSED_LABEL_SIZE)

    ax.set_title("Mediterranean composed Manca basins", fontsize=15, fontweight="bold", pad=10)

    fig.savefig(OUT_COMPOSED, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_COMPOSED}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    plot_atomic()
    plot_composed()


if __name__ == "__main__":
    main()