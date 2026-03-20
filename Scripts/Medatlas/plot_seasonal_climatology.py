#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ============================================================
# PATHS
# ============================================================
IN_DIR = Path("/Users/elena/Projects/IRB/Manca_Argo/Data/Medatlas/Climatology_from_allTS_RR68")
OUT_DIR = Path("/Users/elena/Projects/IRB/Manca_Argo/Plots/Medatlas")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# OPTIONS
# ============================================================
PLOT_STD = False
PLOT_STD_INSET = False
PLOT_T_FROM_TS_SUBSET = True

MIN_N_PLOT = 5
INSET_MAX_DEPTH = 200.0

# display-only smoothing (does NOT change the climatology)
DENSE_NZ_MAIN = 450
DENSE_NZ_INSET = 300

# ============================================================
# STYLE
# ============================================================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

SEASONS = ["JFM", "AMJ", "JAS", "OND"]
SEASON_LABELS = {
    "JFM": "Winter",
    "AMJ": "Spring",
    "JAS": "Summer",
    "OND": "Autumn",
}
SEASON_COLORS = {
    "JFM": "C0",  # blue
    "AMJ": "C2",  # green
    "JAS": "C3",  # red
    "OND": "C1",  # orange
}

# ============================================================
# HELPERS
# ============================================================
def max_depth_from_counts(z, N, min_n=20, min_depth=200.0):
    good = (N >= min_n)
    if not np.any(good):
        return float(min_depth)
    k = np.where(good.any(axis=0))[0].max()
    return max(float(z[k]), float(min_depth))


def dense_profile(z, m, n, zmax, min_n, nz):
    mask = (z >= 0) & (z <= zmax) & np.isfinite(m) & (n >= min_n)
    if mask.sum() < 2:
        return None, None

    zz = z[mask]
    mm = m[mask]
    i = np.argsort(zz)

    zz = zz[i]
    mm = mm[i]

    z_dense = np.linspace(zz.min(), zz.max(), nz)
    m_dense = np.interp(z_dense, zz, mm)
    return z_dense, m_dense


def set_tight_xlim(ax, mean, z, N, zmax, min_n, pad=0.07):
    vals = []
    for i in range(mean.shape[0]):
        mask = (z <= zmax) & np.isfinite(mean[i]) & (N[i] >= min_n)
        if np.any(mask):
            vals.append(mean[i][mask])

    if not vals:
        return

    v = np.concatenate(vals)
    xmin, xmax = np.nanmin(v), np.nanmax(v)
    if xmin == xmax:
        return

    dx = xmax - xmin
    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)


# ============================================================
# PLOTTING
# ============================================================
def plot_panel(ax, z, mean, std, N, title, xlabel, depth_max):

    for i, s in enumerate(SEASONS):
        zd, md = dense_profile(
            z, mean[i], N[i],
            zmax=depth_max,
            min_n=MIN_N_PLOT,
            nz=DENSE_NZ_MAIN,
        )
        if zd is None:
            continue

        ax.plot(md, zd, lw=1.8, color=SEASON_COLORS[s], label=SEASON_LABELS[s])

    ax.set_ylim(depth_max, 0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth [dbar]")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="lower right")

    set_tight_xlim(ax, mean, z, N, depth_max, MIN_N_PLOT)

    # ---- INSET (fixed position, robust) ----
    iax = inset_axes(
        ax,
        width="35%",
        height="35%",
        loc="lower center",
        borderpad=1.2,
    )
    iax.set_facecolor("white")
    iax.patch.set_alpha(1.0)

    for i, s in enumerate(SEASONS):
        zd, md = dense_profile(
            z, mean[i], N[i],
            zmax=INSET_MAX_DEPTH,
            min_n=MIN_N_PLOT,
            nz=DENSE_NZ_INSET,
        )
        if zd is None:
            continue
        iax.plot(md, zd, lw=1.3, color=SEASON_COLORS[s])

    iax.set_ylim(INSET_MAX_DEPTH, 0)
    iax.set_title("0–200", fontsize=10)
    iax.tick_params(labelsize=9)
    iax.grid(True, alpha=0.4)

    set_tight_xlim(iax, mean, z, N, INSET_MAX_DEPTH, MIN_N_PLOT, pad=0.12)


# ============================================================
# MAIN
# ============================================================
def main():
    files = sorted(IN_DIR.glob("MedatlasII_climatology_*_RR68.npz"))
    print(f"Found {len(files)} climatology files")

    for npz in files:
        d = np.load(npz, allow_pickle=True)

        basin = str(d["basin"])
        z = d["z"].astype(float)

        if PLOT_T_FROM_TS_SUBSET and "T_mean_TS" in d.files:
            T_mean, T_std, N_T = d["T_mean_TS"], d["T_std_TS"], d["N_T_TS"]
        else:
            T_mean, T_std, N_T = d["T_mean"], d["T_std"], d["N_T"]

        S_mean, S_std, N_S = d["S_mean"], d["S_std"], d["N_S"]

        depth_max = max(
            max_depth_from_counts(z, N_T),
            max_depth_from_counts(z, N_S),
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), sharey=True)

        plot_panel(
            axes[0], z, T_mean, T_std, N_T,
            title=f"MEDATLAS ({basin}) Temperature",
            xlabel="Temperature [°C]",
            depth_max=depth_max,
        )

        plot_panel(
            axes[1], z, S_mean, S_std, N_S,
            title=f"MEDATLAS ({basin}) Salinity",
            xlabel="Salinity [-]",
            depth_max=depth_max,
        )

        plt.tight_layout()

        ttag = "T_TSsubset" if PLOT_T_FROM_TS_SUBSET else "T_all"
        out = OUT_DIR / f"MedatlasII_{basin}_seasonal_{ttag}_S_RR68.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)

        print(f"Saved {out.name}")

    print("All basins plotted.")


if __name__ == "__main__":
    main()
