#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_figure3_vertical_trend_bars_composed_4_basins_2x2.py

Paper-style 2x2 basin layout for quadratic-fit vertical structure
(1950–2025), for 4 representative Mediterranean composed basins:

a) ALB   b) ION
c) LEV   d) SADR

Within each basin:
    2 rows x 3 columns
    top row    = midpoint linear trend per decade
    bottom row = acceleration per decade^2

Columns:
    Temperature | Salinity | PDA

Colorbar modes
--------------
COLORBAR_MODE = "global"  -> one scale per variable across all basins
COLORBAR_MODE = "basin"   -> each basin gets its own scale
COLORBAR_MODE = "hybrid"  -> T,S global ; PDA basin-specific
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

OUT_DIR = "/Users/elena/Projects/IRB/MedClim/Plots/Publication_Papers"
OUT_NAME = "Fig3_vertical_trend_acceleration_4basins_2x2.png"

BASINS = ["ALB", "ION", "LEV", "SADR"]
BASIN_LABELS = {
    "ALB": "a) ALB",
    "ION": "b) ION",
    "LEV": "c) LEV",
    "SADR": "d) SADR",
}

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

# Significance
ALPHA = 0.05
MASK_NONSIG = False
HATCH_NONSIG = ".."
EDGE_NONSIG = "0.25"

# No-data
NO_DATA_FACE = (0.84, 0.84, 0.84, 1.0)
NO_DATA_HATCH = "///"
NO_DATA_EDGE = "0.35"

# Colormap / scaling
CMAP = "RdBu_r"
ROBUST_Q = 98.0

# Choose:
# "global" / "basin" / "hybrid"
COLORBAR_MODE = "global"

# Figure appearance
FIGSIZE = (11.3, 10.6)
DPI = 300

TITLE_FONTSIZE = 17
PANEL_TITLE_FONTSIZE = 11
BASIN_LABEL_FONTSIZE = 13
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 8
NOTE_FONTSIZE = 11

# Depth / bin geometry
SURFACE_TOP_M = 0.0
MIN_BAR_HEIGHT = 8.0

# Optional depth caps
ZMAX_OVERRIDE = {
    # "ALB": 2000.0,
    # "ION": 4000.0,
    # "LEV": 4000.0,
    # "SADR": 1200.0,
}

# x-range used only as bar width container
XHALF_AUTO_PAD_FRAC = 0.08
XHALF_MIN = 1.0


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


def compute_basin_vlims(df: pd.DataFrame, basins: list[str], vars_: list[str], q: float):
    vlim_tr = {}
    vlim_ac = {}
    for basin in basins:
        vlim_tr[basin] = {}
        vlim_ac[basin] = {}
        for var in vars_:
            d = df[(df["basin"] == basin) & (df["var"] == var)].copy()
            vlim_tr[basin][var] = robust_sym_vlim(d[FIELD_TREND].to_numpy(dtype=float), q=q)
            vlim_ac[basin][var] = robust_sym_vlim(d[FIELD_ACCEL].to_numpy(dtype=float), q=q)
    return vlim_tr, vlim_ac


def get_vlim(mode, basin, var, kind, global_tr, global_ac, basin_tr, basin_ac):
    if mode == "global":
        return global_tr[var] if kind == "trend" else global_ac[var]

    if mode == "basin":
        return basin_tr[basin][var] if kind == "trend" else basin_ac[basin][var]

    if mode == "hybrid":
        if var in ["T", "S"]:
            return global_tr[var] if kind == "trend" else global_ac[var]
        else:
            return basin_tr[basin][var] if kind == "trend" else basin_ac[basin][var]

    raise ValueError(f"Unknown COLORBAR_MODE: {mode}")


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
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
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
            (x0, y0), width, h,
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


def add_colorbar_row(fig, anchor_axes, sm_dict, y, label):
    for j, var in enumerate(VARS):
        pos = anchor_axes[j].get_position()
        cax = fig.add_axes([pos.x0, y, pos.width, 0.012])
        cb = fig.colorbar(sm_dict[var], cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=TICK_FONTSIZE, pad=1)

    fig.text(
        0.5, y - 0.008,
        label,
        ha="center", va="top", fontsize=LABEL_FONTSIZE
    )


# ==========================================================
# MAIN FIGURE
# ==========================================================

def build_paper_figure(
    df: pd.DataFrame,
    basins: list[str],
    out_png: str,
    robust_q: float = 98.0,
    mask_nonsig: bool = False,
):
    missing = [b for b in basins if b not in set(df["basin"].unique())]
    if missing:
        raise ValueError(f"Basins not found in CSV: {missing}")

    global_tr, global_ac = compute_global_vlims(df, basins, VARS, robust_q)
    basin_tr, basin_ac = compute_basin_vlims(df, basins, VARS, robust_q)

    fig = plt.figure(figsize=FIGSIZE, constrained_layout=False)
    outer = fig.add_gridspec(
        nrows=2, ncols=2,
        left=0.09, right=0.985, top=0.885, bottom=0.185,
        wspace=0.12, hspace=0.24
    )

    basin_positions = {
        "ALB": (0, 0),
        "ION": (0, 1),
        "LEV": (1, 0),
        "SADR": (1, 1),
    }

    # for bottom shared colorbars in global/hybrid modes
    global_sm_tr = {}
    global_sm_ac = {}
    basin_bottom_axes = {}

    for basin in basins:
        r, c = basin_positions[basin]
        sub = outer[r, c].subgridspec(2, 3, wspace=0.14, hspace=0.10)

        axs = np.empty((2, 3), dtype=object)

        db = df[df["basin"] == basin].copy()
        if db.empty:
            raise ValueError(f"No data for basin {basin}")

        zmax_auto = float(db["z_m"].max())
        zmax = float(ZMAX_OVERRIDE.get(basin, zmax_auto))

        local_sm_tr = {}
        local_sm_ac = {}

        for irow, kind in enumerate(["trend", "accel"]):
            for j, var in enumerate(VARS):
                ax = fig.add_subplot(sub[irow, j])
                axs[irow, j] = ax

                d = get_basin_var_subset(df, basin, var)

                if kind == "trend":
                    vlim = get_vlim(COLORBAR_MODE, basin, var, "trend",
                                    global_tr, global_ac, basin_tr, basin_ac)
                    sm = draw_bar_column(
                        ax, d,
                        value_col=FIELD_TREND,
                        p_col=FIELD_TREND_P,
                        vlim=vlim,
                        zmax=zmax,
                        mask_nonsig=mask_nonsig
                    )
                    local_sm_tr[var] = sm
                    if basin == "LEV":  # arbitrary holder for shared/global cbars
                        global_sm_tr[var] = ScalarMappable(
                            norm=Normalize(vmin=-vlim, vmax=vlim), cmap=plt.get_cmap(CMAP)
                        )
                        global_sm_tr[var].set_array([])

                else:
                    vlim = get_vlim(COLORBAR_MODE, basin, var, "accel",
                                    global_tr, global_ac, basin_tr, basin_ac)
                    sm = draw_bar_column(
                        ax, d,
                        value_col=FIELD_ACCEL,
                        p_col=FIELD_ACCEL_P,
                        vlim=vlim,
                        zmax=zmax,
                        mask_nonsig=mask_nonsig
                    )
                    local_sm_ac[var] = sm
                    if basin == "LEV":
                        global_sm_ac[var] = ScalarMappable(
                            norm=Normalize(vmin=-vlim, vmax=vlim), cmap=plt.get_cmap(CMAP)
                        )
                        global_sm_ac[var].set_array([])

                if irow == 0:
                    ax.set_title(VAR_TITLES[var], fontsize=PANEL_TITLE_FONTSIZE, pad=8)

                if j == 0:
                    if basin in ["ALB", "LEV"]:
                        ax.set_ylabel("Depth [m]", fontsize=LABEL_FONTSIZE)
                    else:
                        ax.set_ylabel("")
                else:
                    ax.set_yticklabels([])
                if irow == 0:
                    ax.tick_params(labelbottom=False)

        # basin label
        pos = axs[0, 0].get_position()
        fig.text(
            pos.x0 - 0.048,
            pos.y1 + 0.024,
            BASIN_LABELS[basin],
            ha="left", va="center",
            fontsize=BASIN_LABEL_FONTSIZE,
            fontweight="bold"
        )

        # row labels
        pos_tr = axs[0, 0].get_position()
        pos_ac = axs[1, 0].get_position()

        # Row labels only for left-column basin blocks
        if basin in ["ALB", "LEV"]:
            fig.text(
                pos_tr.x0 - 0.060,
                0.5 * (pos_tr.y0 + pos_tr.y1),
                "Linear",
                rotation=90,
                ha="center", va="center",
                fontsize=LABEL_FONTSIZE
            )
            fig.text(
                pos_ac.x0 - 0.060,
                0.5 * (pos_ac.y0 + pos_ac.y1),
                "Accel.",
                rotation=90,
                ha="center", va="center",
                fontsize=LABEL_FONTSIZE
            )

        basin_bottom_axes[basin] = axs[1, :]

        # basin-specific colorbars
        if COLORBAR_MODE == "basin":
            pos_bottom = [axs[1, j].get_position() for j in range(3)]
            y_acc = min(p.y0 for p in pos_bottom) - 0.030
            y_tr = y_acc + 0.035

            add_colorbar_row(fig, axs[1, :], local_sm_tr, y_tr, f"{basin}: Trend (per decade)")
            add_colorbar_row(fig, axs[1, :], local_sm_ac, y_acc, rf"{basin}: Acceleration (per decade$^2$)")

    # Two half-width colorbar groups:
    # left half  -> ALB + LEV
    # right half -> ION + SADR
    if COLORBAR_MODE == "global":
        left_axes = basin_bottom_axes["LEV"]    # bottom-left block
        right_axes = basin_bottom_axes["SADR"]  # bottom-right block

        # vertical positions
        cbar_h = 0.014

        # vertical positions
        y_tr_bar = 0.112
        y_tr_lab = 0.096
        y_ac_bar = 0.050
        y_ac_lab = 0.034
        y_note   = 0.012
        
        # ---------- LEFT HALF ----------
        left_positions = [ax.get_position() for ax in left_axes]
        left_center = 0.5 * (left_positions[0].x0 + left_positions[2].x1)

        for j, var in enumerate(VARS):
            pos = left_positions[j]
            cax = fig.add_axes([pos.x0, y_tr_bar, pos.width, cbar_h])
            cb = fig.colorbar(global_sm_tr[var], cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=TICK_FONTSIZE, pad=1)

        fig.text(
            left_center, y_tr_lab,
            "Trend (per decade)",
            ha="center", va="top", fontsize=LABEL_FONTSIZE
        )

        for j, var in enumerate(VARS):
            pos = left_positions[j]
            cax = fig.add_axes([pos.x0, y_ac_bar, pos.width, cbar_h])
            cb = fig.colorbar(global_sm_ac[var], cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=TICK_FONTSIZE, pad=1)

        fig.text(
            left_center, y_ac_lab,
            r"Acceleration (per decade$^2$)",
            ha="center", va="top", fontsize=LABEL_FONTSIZE
        )

        # ---------- RIGHT HALF ----------
        right_positions = [ax.get_position() for ax in right_axes]
        right_center = 0.5 * (right_positions[0].x0 + right_positions[2].x1)

        for j, var in enumerate(VARS):
            pos = right_positions[j]
            cax = fig.add_axes([pos.x0, y_tr_bar, pos.width, cbar_h])
            cb = fig.colorbar(global_sm_tr[var], cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=TICK_FONTSIZE, pad=1)

        fig.text(
            right_center, y_tr_lab,
            "Trend (per decade)",
            ha="center", va="top", fontsize=LABEL_FONTSIZE
        )

        for j, var in enumerate(VARS):
            pos = right_positions[j]
            cax = fig.add_axes([pos.x0, y_ac_bar, pos.width, cbar_h])
            cb = fig.colorbar(global_sm_ac[var], cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=TICK_FONTSIZE, pad=1)

        fig.text(
            right_center, y_ac_lab,
            r"Acceleration (per decade$^2$)",
            ha="center", va="top", fontsize=LABEL_FONTSIZE
        )

        fig.text(
            0.5, y_note,
            f"Gray hatched = no data; dotted hatch = p ≥ {ALPHA:.2f} | Shared colorbars across all basins",
            ha="center", va="top", fontsize=NOTE_FONTSIZE
        )
    fig.suptitle(
        "Quadratic-fit vertical structure (1950–2025)\n"
        "Representative Mediterranean basins: ALB, ION, LEV, SADR",
        fontsize=TITLE_FONTSIZE,
        y=0.972,
        linespacing=1.05
    )

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote {out_png}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[INFO] Reading:", IN_CSV)
    print("[INFO] Writing to:", os.path.join(OUT_DIR, OUT_NAME))
    print("[INFO] Colorbar mode:", COLORBAR_MODE)

    df = load_data(IN_CSV)

    build_paper_figure(
        df=df,
        basins=BASINS,
        out_png=os.path.join(OUT_DIR, OUT_NAME),
        robust_q=ROBUST_Q,
        mask_nonsig=MASK_NONSIG,
    )

    print("[DONE]")


if __name__ == "__main__":
    main()