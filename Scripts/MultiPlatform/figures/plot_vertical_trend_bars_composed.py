#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_vertical_trend_bars_composed.py

Vertical bar-style profiles of quadratic-fit trend and acceleration
for Mediterranean COMPOSED basins.

For each basin, produce one 2x3 figure:
- Row 1: midpoint linear trend from quadratic fit (per decade)
- Row 2: acceleration term (per decade^2)
- Cols : T, S, SIG0

Display style
-------------
- Colored vertical bars (depth-bin rectangles)
- Depth increases downward
- Zero line shown
- Non-significant bins (p >= ALPHA) are hatched
- No-data bins are shown in light gray with hatch
- Separate robust symmetric color scaling per row+variable
  within each basin figure

Input
-----
quad_trend_accel_fullperiod.csv

Expected columns (written by fit_trends_all_bins.py):
- basin
- z_m
- var
- trend_mid_per_decade
- trend_mid_se_per_decade
- trend_mid_p
- accel_per_decade2
- accel_se_per_decade2
- accel_p
- n_full
- rank_full

Run examples
------------
python plot_vertical_trend_bars_composed.py
python plot_vertical_trend_bars_composed.py --basins WEST ION LEV
python plot_vertical_trend_bars_composed.py --no-mask-nonsig
python plot_vertical_trend_bars_composed.py --robust_q 95

Notes
-----
This script is intended for composed basins first, as discussed.
If later useful, it can easily be extended to atomic basins or to
multi-basin summary panels.
"""

from __future__ import annotations

import os
import argparse
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
OUT_DIR = "/Users/elena/Projects/IRB/MedClim/Plots/MultiPlatform/Trends/VerticalProfiles_Composed"

DEFAULT_BASINS = [
    "ALB", "WEST", "NWM", "TYR", "SIC",
    "NADR", "MADR", "SADR",
    "ION", "LEV", "NAEG", "SAEG",
]

# Variables / layout
VARS = ["T", "S", "SIG0"]
VAR_LABEL = {
    "T": "Temperature trend",
    "S": "Salinity trend",
    "SIG0": "PDA trend",
}
CBAR_LABEL = {
    "T": "T",
    "S": "S",
    "SIG0": "PDA",
}

# Field names from quad_trend_accel_fullperiod.csv
FIELD_TREND = "trend_mid_per_decade"
FIELD_TREND_P = "trend_mid_p"

FIELD_ACCEL = "accel_per_decade2"
FIELD_ACCEL_P = "accel_p"

# Significance
ALPHA = 0.05
MASK_NONSIG = False         # if True: non-sig bins are plotted gray; if False: keep color + hatch
HATCH_NONSIG = ".."
EDGE_NONSIG = "0.25"

# No-data
NO_DATA_FACE = (0.82, 0.82, 0.82, 1.0)
NO_DATA_HATCH = "///"
NO_DATA_EDGE = "0.35"

# Colormap / robust scaling
CMAP = "RdBu_r"
ROBUST_Q = 98.0

# Figure appearance
FIGSIZE = (10.5, 11.0)
DPI = 220

TITLE_FONTSIZE = 14
SUBTITLE_FONTSIZE = 10
LABEL_FONTSIZE = 10
TICK_FONTSIZE = 9

# Depth / bin geometry
SURFACE_TOP_M = 0.0
MIN_BAR_HEIGHT = 8.0   # used if only one bin exists

# Optional depth cap per basin (None = auto full available depth)
ZMAX_OVERRIDE = {
    # "NADR": 100.0,
    # "MADR": 250.0,
}

# Optional x-limits (bar half-width) per variable; None -> auto from data
# This controls only the horizontal plotting width of the bars within each subplot.
XHALF_AUTO_PAD_FRAC = 0.08
XHALF_MIN = 1.0

# Figure spacing
LEFT = 0.10
RIGHT = 0.96
TOP = 0.885
BOTTOM = 0.12
WSPACE = 0.18
HSPACE = 0.36

# Colorbar layout
CBAR_HEIGHT = 0.016
CBAR_PAD = 0.026



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
    """
    Given sorted depth centers z, return top and bottom edges per bin.
    """
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

    # Safety in case of weird binning
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

    if "n_full" in df.columns:
        df["n_full"] = pd.to_numeric(df["n_full"], errors="coerce")
    else:
        df["n_full"] = np.nan

    df = df[np.isfinite(df["z_m"])].copy()
    return df


def get_basin_var_subset(df: pd.DataFrame, basin: str, var: str) -> pd.DataFrame:
    d = df[(df["basin"] == basin) & (df["var"] == var)].copy()
    d = d.sort_values("z_m")
    return d


def draw_bar_column(
    ax,
    d: pd.DataFrame,
    value_col: str,
    p_col: str,
    vlim: float,
    basin: str,
    var: str,
    row_label: str,
    zmax: float | None = None,
    mask_nonsig: bool = False,
):
    cmap = plt.get_cmap(CMAP)
    norm = Normalize(vmin=-vlim, vmax=vlim)

    d = d.sort_values("z_m").copy()
    if zmax is not None:
        d = d[d["z_m"] <= zmax].copy()

    if d.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        return sm

    z = d["z_m"].to_numpy(dtype=float)
    val = d[value_col].to_numpy(dtype=float)
    p = d[p_col].to_numpy(dtype=float) if p_col in d.columns else np.full(len(d), np.nan)

    top, bot = _compute_bin_edges(z)

    # Determine x span from values
    finite_val = val[np.isfinite(val)]
    if finite_val.size == 0:
        xhalf = XHALF_MIN
    else:
        xhalf = nice_symmetric_xlim(vlim)
    x0 = -xhalf
    width = 2.0 * xhalf

    # Draw rectangles
    for i in range(len(z)):
        y0 = top[i]
        h = bot[i] - top[i]

        if np.isfinite(val[i]):
            face = cmap(norm(val[i]))
            hatch = None
            edge = "0.25"
            lw = 0.45

            if np.isfinite(p[i]) and p[i] >= ALPHA:
                hatch = HATCH_NONSIG
                edge = EDGE_NONSIG
                if mask_nonsig:
                    face = NO_DATA_FACE
        else:
            face = NO_DATA_FACE
            hatch = NO_DATA_HATCH
            edge = NO_DATA_EDGE
            lw = 0.50

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

    # Zero line
    ax.axvline(0.0, color="k", lw=1.0, alpha=0.8)

    # Limits / labels
    zmax_here = float(np.nanmax(bot)) if zmax is None else float(zmax)
    ax.set_ylim(zmax_here, 0.0)
    ax.set_xlim(-xhalf, xhalf)

    PANEL_TITLE = {
        "T": "Temperature",
        "S": "Salinity",
        "SIG0": "PDA",
    }

    ax.set_title(PANEL_TITLE.get(var, var), fontsize=SUBTITLE_FONTSIZE)
    ax.grid(False)

    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.set_xlabel(row_label, fontsize=LABEL_FONTSIZE)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm



def add_row_colorbars(fig, axs_row, sm_list, row_text: str):
    fig.canvas.draw()

    positions = [ax.get_position() for ax in axs_row]
    y0 = min(p.y0 for p in positions) - CBAR_PAD - CBAR_HEIGHT

    for ax, sm in zip(axs_row, sm_list):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0, y0, pos.width, CBAR_HEIGHT])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=TICK_FONTSIZE, pad=2)

    x_center = 0.5 * (positions[0].x0 + positions[-1].x1)
    fig.text(
        x_center,
        y0 - 0.030,
        row_text,
        ha="center",
        va="top",
        fontsize=LABEL_FONTSIZE
    )


def build_one_figure(
    df: pd.DataFrame,
    basin: str,
    out_png: str,
    robust_q: float = 98.0,
    mask_nonsig: bool = False,
):
    basin = str(basin).upper().strip()

    # Auto zmax from all vars/rows for this basin
    db = df[df["basin"] == basin].copy()
    if db.empty:
        print(f"[WARN] Basin {basin}: no rows found, skipping.")
        return

    zmax_auto = float(db["z_m"].max())
    zmax = float(ZMAX_OVERRIDE.get(basin, zmax_auto))

    # Robust limits per row+var, basin-specific
    vlim_tr = {}
    vlim_ac = {}
    for var in VARS:
        d = get_basin_var_subset(df, basin, var)
        vlim_tr[var] = robust_sym_vlim(d[FIELD_TREND].to_numpy(dtype=float), q=robust_q)
        vlim_ac[var] = robust_sym_vlim(d[FIELD_ACCEL].to_numpy(dtype=float), q=robust_q)

    fig, axs = plt.subplots(
        2, 3,
        figsize=FIGSIZE,
        constrained_layout=False
    )
    plt.subplots_adjust(
        left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM,
        wspace=WSPACE, hspace=HSPACE
    )

    sm_tr = []
    sm_ac = []

    for j, var in enumerate(VARS):
        d = get_basin_var_subset(df, basin, var)

        sm0 = draw_bar_column(
            axs[0, j],
            d=d,
            value_col=FIELD_TREND,
            p_col=FIELD_TREND_P,
            vlim=vlim_tr[var],
            basin=basin,
            var=var,
            row_label="Trend",
            zmax=zmax,
            mask_nonsig=mask_nonsig,
        )
        sm_tr.append(sm0)

        sm1 = draw_bar_column(
            axs[1, j],
            d=d,
            value_col=FIELD_ACCEL,
            p_col=FIELD_ACCEL_P,
            vlim=vlim_ac[var],
            basin=basin,
            var=var,
            row_label="Acceleration",
            zmax=zmax,
            mask_nonsig=mask_nonsig,
        )
        sm_ac.append(sm1)

    # Y labels only on first column
    axs[0, 0].set_ylabel("Depth [m]", fontsize=LABEL_FONTSIZE)
    axs[1, 0].set_ylabel("Depth [m]", fontsize=LABEL_FONTSIZE)

    for j in [1, 2]:
        axs[0, j].set_yticklabels([])
        axs[1, j].set_yticklabels([])

    # Row labels on left margin
    axs[0, 0].text(
        -0.33, 0.5, "Linear term\n(midpoint trend)",
        transform=axs[0, 0].transAxes,
        rotation=90, ha="center", va="center",
        fontsize=LABEL_FONTSIZE
    )
    axs[1, 0].text(
        -0.33, 0.5, "Acceleration",
        transform=axs[1, 0].transAxes,
        rotation=90, ha="center", va="center",
        fontsize=LABEL_FONTSIZE
    )

    # Figure title
    nsig_note = "non-significant hatched" if not mask_nonsig else "non-significant masked+hatched"
    fig.suptitle(
        f"{basin} — quadratic fit vertical structure (1950–2025)\n"
        f"Top row: midpoint linear trend per decade | Bottom row: acceleration per decade² | {nsig_note}",
        fontsize=TITLE_FONTSIZE,
        y=0.985,
        linespacing=1.15,
    )

    # Row-wise colorbars
    add_row_colorbars(fig, axs[0, :], sm_tr, "Trend (per decade)")
    add_row_colorbars(fig, axs[1, :], sm_ac, "Acceleration (per decade²)")

    # Small legend-like note
    fig.text(
        0.5, 0.018,
        f"Gray hatched = no data; dotted hatch = p ≥ {ALPHA:.2f}",
        ha="center", va="bottom", fontsize=9
    )

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print(f"[OK] Wrote {out_png}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--basins", nargs="*", default=None,
        help="Optional explicit list of basins to plot, e.g. --basins WEST ION LEV"
    )
    ap.add_argument(
        "--robust_q", type=float, default=ROBUST_Q,
        help="Robust symmetric percentile for color scaling (default: 98)"
    )
    ap.add_argument(
        "--mask_nonsig", action="store_true",
        help="If set, non-significant bins are grayed out as well as hatched."
    )
    ap.add_argument(
        "--no-mask-nonsig", action="store_true",
        help="Explicitly keep non-significant bins colored and only hatch them."
    )
    args = ap.parse_args()

    df = load_data(IN_CSV)

    available_basins = set(df["basin"].astype(str).str.upper().str.strip())

    if args.basins and len(args.basins) > 0:
        basins = [b.upper() for b in args.basins if b.upper() in available_basins]
    else:
        basins = [b for b in DEFAULT_BASINS if b in available_basins]

    if len(basins) == 0:
        raise RuntimeError("No basins found to plot.")

    mask_nonsig = MASK_NONSIG
    if args.mask_nonsig:
        mask_nonsig = True
    if args.no_mask_nonsig:
        mask_nonsig = False

    os.makedirs(OUT_DIR, exist_ok=True)

    print("[INFO] Reading:", IN_CSV)
    print("[INFO] Output directory:", OUT_DIR)
    print("[INFO] Basins:", ", ".join(basins))

    for basin in basins:
        out_png = os.path.join(
            OUT_DIR,
            f"{basin}_vertical_trend_accel_bars.png"
        )
        build_one_figure(
            df=df,
            basin=basin,
            out_png=out_png,
            robust_q=float(args.robust_q),
            mask_nonsig=mask_nonsig,
        )

    print("[DONE]")


if __name__ == "__main__":
    main()