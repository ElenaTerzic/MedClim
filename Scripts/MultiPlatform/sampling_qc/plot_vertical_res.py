#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INFILE = Path(str(PROJECT_ROOT / "Stats/Vertical_Sampling/summary_sampling_instrument_ALL_SOURCES.csv"))  # put full path if needed
OUTDIR = Path(str(PROJECT_ROOT / "Stats/Vertical_Sampling/viz_sampling"))
OUTDIR.mkdir(exist_ok=True)

SEASONS = ["JFM","AMJ","JAS","OND"]

def weighted_percentile(values, weights, p):
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[m], w[m]
    if v.size == 0:
        return np.nan
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w) / np.sum(w)
    return float(v[np.searchsorted(cum, p/100.0)])

def wmed(sub, col):
    return weighted_percentile(sub[col], sub["n_profiles"], 50)

df = pd.read_csv(INFILE)

# -------------------------
# 1) SDN instruments counts
# -------------------------
sdn = df[df["source"] == "SDN"].copy()
if not sdn.empty:
    inst = (sdn.groupby("instrument", as_index=False)["n_profiles"]
              .sum()
              .sort_values("n_profiles", ascending=False)
              .head(30))

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    ax.bar(range(len(inst)), inst["n_profiles"].values)
    ax.set_xticks(range(len(inst)))
    ax.set_xticklabels(inst["instrument"].fillna(""), rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Total profiles (sum over basins+seasons)")
    ax.set_title("SDN: Top instruments by profile count")
    fig.tight_layout()
    fig.savefig(OUTDIR / "sdn_top_instruments_counts.png", dpi=200)
    plt.close(fig)

    # ---------------------------------------------------
    # 2) SDN instruments: dz0_200 vs zmax (typical values)
    # ---------------------------------------------------
    g = (sdn.groupby("instrument", as_index=False)
            .agg(n_profiles_total=("n_profiles","sum"),
                 med_dz0_200=("median_dz_0_200","median"),
                 med_zmax=("median_zmax","median"),
                 med_nlevels=("median_nlevels","median")))

    g = g[(g["n_profiles_total"] > 200) & np.isfinite(g["med_dz0_200"]) & np.isfinite(g["med_zmax"])]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.scatter(g["med_dz0_200"], g["med_zmax"],
               s=np.clip(g["n_profiles_total"]/200, 20, 600),
               alpha=0.7)
    for _, r in g.iterrows():
        ax.text(r["med_dz0_200"], r["med_zmax"], str(r["instrument"])[:18], fontsize=7)

    ax.set_xlabel("Median dz (0–200 m) [m]")
    ax.set_ylabel("Median zmax [m]")
    ax.set_title("SDN instruments: typical spacing vs typical depth\n(size ~ total profiles)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "sdn_instruments_scatter_dz_zmax.png", dpi=200)
    plt.close(fig)

# ---------------------------------------------------------
# 3) Heatmap per source: basin x season weighted-median dz0_200
# ---------------------------------------------------------
for src in sorted(df["source"].dropna().unique()):
    d = df[df["source"] == src].copy()
    if d.empty:
        continue

    # aggregate across instruments within each basin/season
    agg = (d.groupby(["basin","season"], as_index=False)
             .apply(lambda sub: pd.Series({
                 "n_profiles": sub["n_profiles"].sum(),
                 "wmed_dz0_200": wmed(sub, "median_dz_0_200"),
             }))
             .reset_index(drop=True))

    basins = sorted(agg["basin"].dropna().unique())
    if not basins:
        continue

    mat = np.full((len(basins), len(SEASONS)), np.nan)
    nmat = np.zeros((len(basins), len(SEASONS)), float)

    for i, b in enumerate(basins):
        for j, s in enumerate(SEASONS):
            row = agg[(agg["basin"] == b) & (agg["season"] == s)]
            if len(row):
                mat[i,j] = row["wmed_dz0_200"].values[0]
                nmat[i,j] = row["n_profiles"].values[0]

    fig = plt.figure(figsize=(10, max(5, 0.25*len(basins)+2)))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(f"{src}: weighted-median dz (0–200 m)\n(numbers are n_profiles")
    ax.set_xticks(range(len(SEASONS)))
    ax.set_xticklabels(SEASONS)
    ax.set_yticks(range(len(basins)))
    ax.set_yticklabels(basins, fontsize=8)

    for i in range(len(basins)):
        for j in range(len(SEASONS)):
            if np.isfinite(mat[i,j]):
                ax.text(j, i, f"{int(nmat[i,j])}", ha="center", va="center", fontsize=6)

    fig.colorbar(im, ax=ax, shrink=0.8, label="dz [m] (weighted median)")
    fig.tight_layout()
    fig.savefig(OUTDIR / f"heatmap_{src}_dz0_200.png", dpi=200)
    plt.close(fig)

print("Wrote plots to:", OUTDIR.resolve())
