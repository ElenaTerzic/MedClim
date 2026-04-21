#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ======================
# CONFIG
# ======================

IN_CSV = Path("/Users/elena/Projects/IRB/MedClim/Stats/cast_counts_bybin_by_platform_period_T_composed_annualPooled.csv")
OUTDIR = Path("/Users/elena/Projects/IRB/MedClim/Plots/MultiPlatform/CastCounts_AnnualPooled_Clean")
OUTDIR.mkdir(parents=True, exist_ok=True)

BASINS = ["NWM", "ION", "LEV"]

PERIODS = {
    "21c_2000_2025": 26,
    "20c_1950_1999": 50,
}

PLOT_ZMAX = 2000

SDN_CLASSES = ["SDN_CTD", "SDN_MBTXBT", "SDN_Bottle"]
ORDER = ["SDN_CTD", "SDN_MBTXBT", "SDN_Bottle", "ARGO"]

COLORS = {
    "SDN_CTD": "#d62728",
    "SDN_MBTXBT": "#ff7f0e",
    "SDN_Bottle": "#2ca02c",
    "ARGO": "0.2",
}

LABELS = {
    "SDN_CTD": "SDN CTD",
    "SDN_MBTXBT": "SDN MBT/XBT",
    "SDN_Bottle": "SDN Bottle",
    "ARGO": "ARGO",
}

FIGSIZE = (16, 13)
SUPTITLE_FS = 16
TITLE_FS = 13
LABEL_FS = 11
TICK_FS = 9
LEGEND_FS = 10

GRID_ALPHA = 0.08
BAR_ALPHA = 0.95
ARGO_FRACTION_ALPHA = 0.5
ARGO_MARKER_SIZE = 12
ARGO_MARKER_ALPHA = 0.8


# ======================
# LOAD + PREP
# ======================

df = pd.read_csv(IN_CSV)

df["basin"] = df["basin"].str.upper()
df = df[df["basin"].isin(BASINS)]
df = df[df["z_m"] <= PLOT_ZMAX]

def classify(row):
    if row["source"] == "SDN":
        return row["sdn_class"]
    return row["source"]

df["cat"] = df.apply(classify, axis=1)
df = df[df["cat"].isin(ORDER)]
df["casts_per_year"] = df["n_casts"] / df["period"].map(PERIODS)


# ======================
# HELPERS
# ======================

def get_depths(dfb):
    return np.array(sorted(dfb["z_m"].unique()))

def frac_table(sub, depths):
    rows = []
    for z in depths:
        zz = sub[sub["z_m"] == z]
        total = zz["n_casts"].sum()
        row = {"z_m": z}
        for c in ORDER:
            val = zz.loc[zz["cat"] == c, "n_casts"].sum()
            row[c] = val / total if total > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def get_global_xlim(df):
    xmax = 0
    for basin in BASINS:
        for p in PERIODS:
            sub = df[(df["basin"] == basin) & (df["period"] == p)]
            depths = get_depths(sub)
            if len(depths) == 0:
                continue

            total = np.zeros_like(depths, dtype=float)
            for cls in SDN_CLASSES:
                s = sub[sub["cat"] == cls]
                m = dict(zip(s["z_m"], s["casts_per_year"]))
                vals = np.array([m.get(z, 0) for z in depths])
                total += vals

            if len(total):
                xmax = max(xmax, total.max())

            ar = sub[sub["cat"] == "ARGO"]
            if not ar.empty:
                m = dict(zip(ar["z_m"], ar["casts_per_year"]))
                vals = np.array([m.get(z, np.nan) for z in depths])
                if np.isfinite(vals).any():
                    xmax = max(xmax, np.nanmax(vals))

    return xmax * 1.1


# ======================
# PLOT
# ======================

fig, axes = plt.subplots(3, 4, figsize=FIGSIZE, sharey=True)
xmax = get_global_xlim(df)

# subtle column shading to distinguish centuries
for col in [0, 2]:  # 21c columns
    for i in range(3):
        axes[i, col].set_facecolor("#f7f7f7")
for col in [1, 3]:  # 20c columns
    for i in range(3):
        axes[i, col].set_facecolor("#ffffff")

for i, basin in enumerate(BASINS):
    dfb = df[df["basin"] == basin]
    depths = get_depths(dfb)

    if depths.size > 1:
        dz = np.median(np.diff(depths))
        bar_height = 0.78 * dz
    else:
        bar_height = 15.0

    for j, period in enumerate(["21c_2000_2025", "20c_1950_1999"]):
        sub = dfb[dfb["period"] == period]

        # ---------------- COUNTS ----------------
        ax = axes[i, j]
        left = np.zeros_like(depths, dtype=float)

        for cls in SDN_CLASSES:
            s = sub[sub["cat"] == cls]
            m = dict(zip(s["z_m"], s["casts_per_year"]))
            vals = np.array([m.get(z, 0) for z in depths])

            ax.barh(
                depths,
                vals,
                left=left,
                height=bar_height,
                color=COLORS[cls],
                edgecolor="none",
                alpha=BAR_ALPHA
            )
            left += vals

        ar = sub[sub["cat"] == "ARGO"]
        if not ar.empty:
            m = dict(zip(ar["z_m"], ar["casts_per_year"]))
            x = np.array([m.get(z, np.nan) for z in depths])
            ax.scatter(
                x, depths,
                s=ARGO_MARKER_SIZE,
                color=COLORS["ARGO"],
                alpha=ARGO_MARKER_ALPHA,
                zorder=5
            )

        ax.set_xlim(0, xmax)
        ax.set_ylim(PLOT_ZMAX, 0)
        ax.grid(alpha=GRID_ALPHA)
        ax.tick_params(labelsize=TICK_FS)

        if i == 2:
            ax.set_xlabel("casts per year", fontsize=LABEL_FS)
        if j == 0:
            ax.set_ylabel(f"{basin}\nDepth (m)", fontsize=LABEL_FS)

        # ---------------- FRACTIONS ----------------
        axf = axes[i, j + 2]
        ft = frac_table(sub, depths)
        left = np.zeros(len(ft), dtype=float)

        for cls in ORDER:
            vals = ft[cls].values
            alpha_here = ARGO_FRACTION_ALPHA if cls == "ARGO" else BAR_ALPHA

            axf.barh(
                ft["z_m"],
                vals,
                left=left,
                height=bar_height,
                color=COLORS[cls],
                edgecolor="none",
                alpha=alpha_here
            )
            left += np.nan_to_num(vals)

        axf.set_xlim(0, 1)
        axf.set_ylim(PLOT_ZMAX, 0)
        axf.grid(alpha=GRID_ALPHA)
        axf.tick_params(labelsize=TICK_FS)

        if i == 2:
            axf.set_xlabel("fraction", fontsize=LABEL_FS)

# column titles
axes[0, 0].set_title("21st century (2000–2025)\ncounts per year", fontsize=TITLE_FS)
axes[0, 1].set_title("20th century (1950–1999)\ncounts per year", fontsize=TITLE_FS)
axes[0, 2].set_title("21st century\nplatform fraction", fontsize=TITLE_FS)
axes[0, 3].set_title("20th century\nplatform fraction", fontsize=TITLE_FS)

# legend
handles = [
    Patch(facecolor=COLORS[c], label=LABELS[c])
    for c in SDN_CLASSES
]
handles.append(
    plt.Line2D([0], [0], marker="o", color="none",
               markerfacecolor=COLORS["ARGO"],
               markeredgecolor="none",
               label="ARGO")
)

fig.legend(
    handles=handles,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.965),  # slightly lower than before
    fontsize=LEGEND_FS,
    handlelength=1.6,
    columnspacing=1.5
)

fig.suptitle("Sampling support across representative basins", fontsize=SUPTITLE_FS, y=0.988)

plt.tight_layout(rect=[0, 0, 1, 0.955])

# save
out_png = OUTDIR / "sampling_support_3x4_final.png"
out_pdf = OUTDIR / "sampling_support_3x4_final.pdf"

fig.savefig(out_png, dpi=250)
fig.savefig(out_pdf)
plt.close(fig)

print("Saved:", out_png)
print("Saved:", out_pdf)