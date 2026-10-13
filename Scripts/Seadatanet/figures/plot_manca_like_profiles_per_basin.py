#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reads : Data/Seadatanet/climatology_sdn_{20c_1950_1999,21c_2000_2019}_after_1sigma.csv
# Writes: Plots/Seadatanet/Seasonal_Profiles_Per_Basin/<period>/<basin>_seasonal_profiles_TS_<tag>.png

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Scripts.paths import DATA, PLOTS

# -----------------------------
# INPUTS (BOTH PERIODS)
# -----------------------------
INPUTS = [
    (
        "20c_1950_1999",
        DATA / "Seadatanet" / "climatology_sdn_20c_1950_1999_after_1sigma.csv",
    ),
    (
        "21c_2000_2019",
        DATA / "Seadatanet" / "climatology_sdn_21c_2000_2019_after_1sigma.csv",
    ),
]

OUTBASE = PLOTS / "Seadatanet" / "Seasonal_Profiles_Per_Basin"
OUTBASE.mkdir(parents=True, exist_ok=True)

# -----------------------------
# CONFIG
# -----------------------------
INSET_ZMAX = 200.0

MIN_N_FOR_ZMAX = 10
MAX_PLOT_Z = 2000.0
MIN_PLOT_Z = 200.0

SEASON_MAP = {
    "JFM": "Winter",
    "AMJ": "Spring",
    "JAS": "Summer",
    "OND": "Autumn",
    "WINTER": "Winter",
    "SPRING": "Spring",
    "SUMMER": "Summer",
    "AUTUMN": "Autumn",
}
SEASONS_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

COLORS = {
    "Winter": "tab:blue",
    "Spring": "tab:green",
    "Summer": "tab:red",
    "Autumn": "tab:orange",
}

INSET_CFG = {
    "T": dict(width="38%", height="38%", loc="lower left", bbox_to_anchor=(0.18, 0.12, 1, 1)),
    "S": dict(width="38%", height="38%", loc="lower left", bbox_to_anchor=(0.16, 0.12, 1, 1)),
}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

# -----------------------------
# HELPERS
# -----------------------------
def _std_shade(ax, x, y, sd, color, alpha=0.15):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(sd)
    if m.any():
        ax.fill_betweenx(y[m], (x - sd)[m], (x + sd)[m], color=color, alpha=alpha, linewidth=0)

def _add_inset(parent_ax, var_key: str):
    cfg = INSET_CFG[var_key]
    return inset_axes(
        parent_ax,
        width=cfg["width"],
        height=cfg["height"],
        loc=cfg["loc"],
        bbox_to_anchor=cfg["bbox_to_anchor"],
        bbox_transform=parent_ax.transAxes,
        borderpad=0.8,
    )

def compute_basin_zmax(df_basin: pd.DataFrame) -> float:
    g = df_basin.copy()
    g["ok"] = (g["n_T"] >= MIN_N_FOR_ZMAX) | (g["n_S"] >= MIN_N_FOR_ZMAX)
    depths_ok = g.loc[g["ok"], "z_m"].to_numpy(float)
    if depths_ok.size == 0:
        return MIN_PLOT_Z
    zmax = float(np.nanmax(depths_ok))
    zmax = max(MIN_PLOT_Z, min(MAX_PLOT_Z, zmax))
    return zmax

def plot_one_basin(df_basin: pd.DataFrame, basin: str, tag: str, outdir: Path):
    zmax = compute_basin_zmax(df_basin)

    fig, (axT, axS) = plt.subplots(1, 2, figsize=(12.5, 5.5), sharey=True)
    fig.suptitle(f"{basin}: Seasonal climatology ({tag}) (mean ± std on main panels)")

    for ax in (axT, axS):
        ax.set_ylim(zmax, 0)
        ax.grid(True, alpha=0.35)
        ax.set_ylabel("Depth [m]")

    axT.set_title("Temperature [°C]")
    axS.set_title("Salinity [-]")
    axT.set_xlabel("Temperature [°C]")
    axS.set_xlabel("Salinity [-]")

    iaT = _add_inset(axT, "T")
    iaS = _add_inset(axS, "S")
    for iax in (iaT, iaS):
        iax.set_ylim(INSET_ZMAX, 0)
        iax.grid(True, alpha=0.35)
        iax.tick_params(labelsize=10)
    iaT.set_title(f"0–{int(INSET_ZMAX)} m", fontsize=11)
    iaS.set_title(f"0–{int(INSET_ZMAX)} m", fontsize=11)

    any_plotted = False

    for season in SEASONS_ORDER:
        g = df_basin[df_basin["season_name"] == season].sort_values("z_m")
        if g.empty:
            continue

        z = g["z_m"].to_numpy(float)

        Tm = g["T_mean"].to_numpy(float)
        Ts = g["T_std"].to_numpy(float)
        mT = np.isfinite(z) & np.isfinite(Tm) & (z >= 0) & (z <= zmax)
        if mT.any():
            axT.plot(Tm[mT], z[mT], color=COLORS[season], lw=2.0, label=season)
            _std_shade(axT, Tm[mT], z[mT], Ts[mT], COLORS[season])
            any_plotted = True

        Sm = g["S_mean"].to_numpy(float)
        Ss = g["S_std"].to_numpy(float)
        mS = np.isfinite(z) & np.isfinite(Sm) & (z >= 0) & (z <= zmax)
        if mS.any():
            axS.plot(Sm[mS], z[mS], color=COLORS[season], lw=2.0, label=season)
            _std_shade(axS, Sm[mS], z[mS], Ss[mS], COLORS[season])
            any_plotted = True

        mi = np.isfinite(z) & (z >= 0) & (z <= INSET_ZMAX)
        if mi.any():
            if np.isfinite(Tm[mi]).any():
                iaT.plot(Tm[mi], z[mi], color=COLORS[season], lw=1.6)
            if np.isfinite(Sm[mi]).any():
                iaS.plot(Sm[mi], z[mi], color=COLORS[season], lw=1.6)

    if any_plotted:
        axS.legend(loc="lower right", frameon=True)

    #axT.text(0.02, 0.02, f"auto zmax={int(zmax)} m (n≥{MIN_N_FOR_ZMAX})",
     #        transform=axT.transAxes, fontsize=10, va="bottom")

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    outpng = outdir / f"{basin}_seasonal_profiles_TS_{tag}.png"
    fig.savefig(outpng, dpi=200)
    plt.close(fig)
    print("Wrote:", outpng)

def load_and_clean(infile: Path) -> pd.DataFrame:
    df = pd.read_csv(infile)

    need = {"basin", "season", "z_m", "n_T", "T_mean", "T_std", "n_S", "S_mean", "S_std"}
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise SystemExit(f"Missing columns in {infile}:\n{missing}")

    df["season"] = df["season"].astype(str).str.strip()
    df["season_name"] = df["season"].str.upper().map(SEASON_MAP)

    df["z_m"] = pd.to_numeric(df["z_m"], errors="coerce")
    for c in ["n_T", "n_S"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ["T_mean", "T_std", "S_mean", "S_std"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["basin", "season_name", "z_m"])
    return df

def main():
    for tag, infile in INPUTS:
        if not infile.exists():
            print(f"WARNING: missing input: {infile}")
            continue

        outdir = OUTBASE / tag
        outdir.mkdir(parents=True, exist_ok=True)

        print("\n====================================")
        print("Plotting:", tag)
        print("Input:   ", infile)
        print("Outdir:  ", outdir)
        print("====================================")

        df = load_and_clean(infile)
        basins = sorted(df["basin"].astype(str).unique().tolist())
        print(f"Basins found: {len(basins)}")

        for basin in basins:
            plot_one_basin(df[df["basin"] == basin].copy(), basin, tag, outdir)

if __name__ == "__main__":
    main()
