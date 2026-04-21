#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supplementary composite figures for quadratic-fit vertical structure (1950–2025)
across additional Mediterranean composed basins not shown in the main Figure 4.

Creates 3 composite figures (each: 3 basins x 2 rows x 3 variables):
    SFIG_A = NADR, MADR, SADR
    SFIG_B = SIC, NAEG, SAEG
    SFIG_C = ALB, WEST, TYR

Notes
-----
- Main Figure 4 already shows NWM, ION, and LEV.
- Same colorbar limits are used across all 3 basins within each composite figure
  (COLORBAR_MODE = "global").
- For NADR, the deepest bin is removed before plotting to avoid a poorly sampled,
  non-representative deepest level.
- Output is written as both PNG and PDF.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ==========================================================
# USER CONFIG
# ==========================================================

IN_CSV = "/Users/elena/Projects/IRB/MedClim/Results/Trends/Composed/quad_trend_accel_fullperiod_composed.csv"
OUT_DIR = "/Users/elena/Projects/IRB/MedClim/Plots/Supplement"

SUPP_FIGURES = [
    {
        "name": "Supp_vertical_trend_acceleration_NADR_MADR_SADR",
        "basins": ["NADR", "MADR", "SADR"],
        "labels": {"NADR": "a) NADR", "MADR": "b) MADR", "SADR": "c) SADR"},
    },
    {
        "name": "Supp_vertical_trend_acceleration_SIC_NAEG_SAEG",
        "basins": ["SIC", "NAEG", "SAEG"],
        "labels": {"SIC": "a) SIC", "NAEG": "b) NAEG", "SAEG": "c) SAEG"},
    },
    {
        "name": "Supp_vertical_trend_acceleration_ALB_WEST_TYR",
        "basins": ["ALB", "WEST", "TYR"],
        "labels": {"ALB": "a) ALB", "WEST": "b) WEST", "TYR": "c) TYR"},
    },
]


VARS = ["T", "S", "SIG0"]
VAR_TITLES = {
    "T": "Temperature",
    "S": "Salinity",
    "SIG0": "PDA",
}

FIELD_TREND = "trend_mid_per_decade"
FIELD_TREND_P = "trend_mid_p"
FIELD_ACCEL = "accel_per_decade2"
FIELD_ACCEL_P = "accel_p"

ALPHA = 0.05
MASK_NONSIG = False
HATCH_NONSIG = ".."
EDGE_NONSIG = "0.25"

NO_DATA_FACE = (0.84, 0.84, 0.84, 1.0)
NO_DATA_HATCH = "///"
NO_DATA_EDGE = "0.35"

CMAP = "RdBu_r"
ROBUST_Q = 98.0
COLORBAR_MODE = "global"   # keep same cbar limits within each 3-basin composite

FIGSIZE = (13.8, 8.6)
DPI = 300

PANEL_TITLE_FONTSIZE = 11
BASIN_LABEL_FONTSIZE = 14
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 8
NOTE_FONTSIZE = 10

SURFACE_TOP_M = 0.0
MIN_BAR_HEIGHT = 8.0
XHALF_AUTO_PAD_FRAC = 0.08
XHALF_MIN = 1.0

# Optional depth caps
ZMAX_OVERRIDE = {
    # "WEST": 3000.0,
    # "TYR": 3500.0,
    # "SIC": 1750.0,
    # "ALB": 2000.0,
    # "NADR": 80.0,
    # "ION": 4000.0,
    # "MADR": 250.0,
    # "SADR": 1200.0,
    # "NAEG": 1500.0,
}

# Remove specific problematic bins (exact z_m values can be listed).
# For NADR we drop the deepest bin (~100 m in your analysis output).
DROP_Z_BY_BASIN = {
    "NADR": [100.0],
}
DROP_Z_TOL = 1.0  # metres tolerance around listed z values


# ==========================================================
# HELPERS
# ==========================================================

def robust_sym_vlim(values: np.ndarray, q: float = 98.0) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    s = float(np.nanpercentile(np.abs(v), q))
    if not np.isfinite(s) or s <= 0:
        return 1.0
    return s


def nice_symmetric_xlim(vlim: float) -> float:
    if not np.isfinite(vlim) or vlim <= 0:
        return XHALF_MIN

    candidates = np.array([
        0.01, 0.02, 0.05,
        0.1, 0.2, 0.5,
        1, 2, 5, 10, 20, 50, 100
    ], dtype=float)

    target = vlim * (1.0 + XHALF_AUTO_PAD_FRAC)
    idx = np.searchsorted(candidates, target, side="left")
    if idx >= len(candidates):
        return float(target)
    return float(max(candidates[idx], XHALF_MIN))


def _compute_bin_edges(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=float)
    n = len(z)

    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if n == 1:
        h = max(MIN_BAR_HEIGHT, z[0] if z[0] > 0 else MIN_BAR_HEIGHT)
        top = np.array([max(SURFACE_TOP_M, z[0] - 0.5 * h)], dtype=float)
        bot = np.array([z[0] + 0.5 * h], dtype=float)
        return top, bot

    mids = 0.5 * (z[:-1] + z[1:])
    top = np.empty(n, dtype=float)
    bot = np.empty(n, dtype=float)

    top[0] = SURFACE_TOP_M
    bot[-1] = z[-1] + 0.5 * (z[-1] - z[-2])

    top[1:] = mids
    bot[:-1] = mids

    bad = bot <= top
    bot[bad] = top[bad] + MIN_BAR_HEIGHT

    return top, bot


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    needed = [
        "basin", "z_m", "var",
        FIELD_TREND, FIELD_TREND_P,
        FIELD_ACCEL, FIELD_ACCEL_P,
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["basin"] = df["basin"].astype(str).str.upper().str.strip()
    df["var"] = df["var"].astype(str).str.upper().str.strip()
    df["z_m"] = pd.to_numeric(df["z_m"], errors="coerce")

    for c in [FIELD_TREND, FIELD_TREND_P, FIELD_ACCEL, FIELD_ACCEL_P]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[np.isfinite(df["z_m"])].copy()
    return df


def apply_basin_bin_drops(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    keep = np.ones(len(out), dtype=bool)

    for basin, z_list in DROP_Z_BY_BASIN.items():
        basin_mask = out["basin"].eq(basin)
        for z0 in z_list:
            basin_mask_this = basin_mask & np.isclose(
                out["z_m"].to_numpy(dtype=float),
                z0,
                atol=DROP_Z_TOL
            )
            keep &= ~basin_mask_this

    return out.loc[keep].copy()


def get_basin_var_subset(df: pd.DataFrame, basin: str, var: str) -> pd.DataFrame:
    d = df[(df["basin"] == basin) & (df["var"] == var)].copy()
    d = d.sort_values("z_m")
    return d


def compute_global_vlims(df: pd.DataFrame, basins: list[str], vars_: list[str], q: float):
    vlim_tr = {}
    vlim_ac = {}

    for var in vars_:
        dvar = df[(df["basin"].isin(basins)) & (df["var"] == var)].copy()
        vlim_tr[var] = robust_sym_vlim(dvar[FIELD_TREND].to_numpy(dtype=float), q=q)
        vlim_ac[var] = robust_sym_vlim(dvar[FIELD_ACCEL].to_numpy(dtype=float), q=q)

    return vlim_tr, vlim_ac


def draw_bar_column(
    ax,
    d: pd.DataFrame,
    value_col: str,
    p_col: str,
    vlim: float,
    zmax: float | None = None,
    mask_nonsig: bool = False,
):
    cmap = plt.get_cmap(CMAP)
    norm = Normalize(vmin=-vlim, vmax=vlim)

    d = d.sort_values("z_m").copy()
    if zmax is not None:
        d = d[d["z_m"] <= zmax].copy()

    if d.empty:
        ax.text(
            0.5, 0.5, "No data",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=8
        )
        ax.set_axis_off()
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        return sm

    z = d["z_m"].to_numpy(dtype=float)
    val = d[value_col].to_numpy(dtype=float)
    p = d[p_col].to_numpy(dtype=float) if p_col in d.columns else np.full(len(d), np.nan)

    top, bot = _compute_bin_edges(z)

    xhalf = nice_symmetric_xlim(vlim)
    x0 = -xhalf
    width = 2.0 * xhalf

    for i in range(len(z)):
        y0 = top[i]
        h = bot[i] - top[i]

        if np.isfinite(val[i]):
            face = cmap(norm(val[i]))
            hatch = None
            edge = "0.25"
            lw = 0.40

            if np.isfinite(p[i]) and p[i] >= ALPHA:
                hatch = HATCH_NONSIG
                edge = EDGE_NONSIG
                if mask_nonsig:
                    face = NO_DATA_FACE
        else:
            face = NO_DATA_FACE
            hatch = NO_DATA_HATCH
            edge = NO_DATA_EDGE
            lw = 0.45

        rect = Rectangle(
            (x0, y0),
            width,
            h,
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            hatch=hatch,
        )
        ax.add_patch(rect)

    ax.axvline(0.0, color="k", lw=0.9, alpha=0.85)

    zmax_here = float(np.nanmax(bot)) if zmax is None else float(zmax)
    ax.set_ylim(zmax_here, 0.0)
    ax.set_xlim(-xhalf, xhalf)

    ax.grid(False)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


# ==========================================================
# FIGURE BUILDER
# ==========================================================

def build_composite_figure(
    df: pd.DataFrame,
    basins: list[str],
    basin_labels: dict[str, str],
    out_png: str,
    robust_q: float = 98.0,
    mask_nonsig: bool = False,
):
    missing = [b for b in basins if b not in set(df["basin"].unique())]
    if missing:
        raise ValueError(f"Basins not found in CSV: {missing}")

    global_tr, global_ac = compute_global_vlims(df, basins, VARS, robust_q)

    fig = plt.figure(figsize=FIGSIZE, constrained_layout=False)
    outer = fig.add_gridspec(
        nrows=1,
        ncols=3,
        left=0.075,
        right=0.985,
        top=0.86,
        bottom=0.24,
        wspace=0.14
    )

    basin_axes = {}

    for ibasin, basin in enumerate(basins):
        sub = outer[0, ibasin].subgridspec(2, 3, wspace=0.16, hspace=0.10)
        axs = np.empty((2, 3), dtype=object)

        db = df[df["basin"] == basin].copy()
        if db.empty:
            raise ValueError(f"No data for basin {basin}")

        zmax_auto = float(db["z_m"].max())
        zmax = float(ZMAX_OVERRIDE.get(basin, zmax_auto))

        for irow, kind in enumerate(["trend", "accel"]):
            for j, var in enumerate(VARS):
                ax = fig.add_subplot(sub[irow, j])
                axs[irow, j] = ax

                d = get_basin_var_subset(df, basin, var)
                vlim = global_tr[var] if kind == "trend" else global_ac[var]

                draw_bar_column(
                    ax=ax,
                    d=d,
                    value_col=FIELD_TREND if kind == "trend" else FIELD_ACCEL,
                    p_col=FIELD_TREND_P if kind == "trend" else FIELD_ACCEL_P,
                    vlim=vlim,
                    zmax=zmax,
                    mask_nonsig=mask_nonsig,
                )

                if irow == 0:
                    ax.set_title(VAR_TITLES[var], fontsize=PANEL_TITLE_FONTSIZE, pad=10)

                if j == 0:
                    if ibasin == 0:
                        ax.set_ylabel("Depth [m]", fontsize=LABEL_FONTSIZE)
                    else:
                        ax.set_ylabel("")
                else:
                    ax.set_yticklabels([])

                ax.set_xticks([])
                ax.tick_params(axis="x", bottom=False, labelbottom=False)

        basin_axes[basin] = axs

        pos = axs[0, 0].get_position()
        fig.text(
            pos.x0 - 0.050,
            pos.y1 + 0.040,
            basin_labels[basin],
            ha="left",
            va="center",
            fontsize=BASIN_LABEL_FONTSIZE,
            fontweight="bold"
        )

    top_left = basin_axes[basins[0]][0, 0].get_position()
    bot_left = basin_axes[basins[0]][1, 0].get_position()

    fig.text(
        top_left.x0 - 0.055,
        0.5 * (top_left.y0 + top_left.y1),
        "Linear",
        rotation=90,
        ha="center",
        va="center",
        fontsize=LABEL_FONTSIZE
    )
    fig.text(
        bot_left.x0 - 0.055,
        0.5 * (bot_left.y0 + bot_left.y1),
        "Accel.",
        rotation=90,
        ha="center",
        va="center",
        fontsize=LABEL_FONTSIZE
    )

    trend_bar_y = 0.205
    accel_bar_y = 0.145
    cbar_h = 0.016
    note_y = 0.090

    for basin in basins:
        axs = basin_axes[basin]
        x0 = axs[1, 0].get_position().x0
        x1 = axs[1, 2].get_position().x1
        width = x1 - x0

        sm_tr = ScalarMappable(
            norm=Normalize(vmin=-global_tr["T"], vmax=global_tr["T"]),
            cmap=plt.get_cmap(CMAP)
        )
        sm_tr.set_array([])

        sm_ac = ScalarMappable(
            norm=Normalize(vmin=-global_ac["T"], vmax=global_ac["T"]),
            cmap=plt.get_cmap(CMAP)
        )
        sm_ac.set_array([])

        cax1 = fig.add_axes([x0, trend_bar_y, width, cbar_h])
        cb1 = fig.colorbar(sm_tr, cax=cax1, orientation="horizontal")
        cb1.ax.tick_params(labelsize=TICK_FONTSIZE, pad=1)
        cb1.set_label("Trend (per decade)", fontsize=LABEL_FONTSIZE, labelpad=2)

        cax2 = fig.add_axes([x0, accel_bar_y, width, cbar_h])
        cb2 = fig.colorbar(sm_ac, cax=cax2, orientation="horizontal")
        cb2.ax.tick_params(labelsize=TICK_FONTSIZE, pad=1)
        cb2.set_label(r"Acceleration (per decade$^2$)", fontsize=LABEL_FONTSIZE, labelpad=2)

    fig.text(
        0.5,
        note_y,
        f"Gray hatched = no data; dotted hatch = p ≥ {ALPHA:.2f}",
        ha="center",
        va="center",
        fontsize=NOTE_FONTSIZE
    )

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote {out_png}")
    print(f"[OK] Wrote {out_png.replace('.png', '.pdf')}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[INFO] Reading:", IN_CSV)
    print("[INFO] Writing to:", OUT_DIR)

    df = load_data(IN_CSV)
    df = apply_basin_bin_drops(df)

    for spec in SUPP_FIGURES:
        out_png = os.path.join(OUT_DIR, spec["name"] + ".png")

        print(f"[INFO] Building: {spec['name']} -> {spec['basins']}")

        build_composite_figure(
            df=df,
            basins=spec["basins"],
            basin_labels=spec["labels"],
            out_png=out_png,
            robust_q=ROBUST_Q,
            mask_nonsig=MASK_NONSIG,
        )

    print("[DONE]")


if __name__ == "__main__":
    main()