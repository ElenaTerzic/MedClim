#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
paper_figure_ssl_6panels.py

Creates ONE paper-style figure (2x3 panels):

Top row:
 A) SSL occurrence (%) per year (overall; optional per-source faint lines)
 B) Seasonal SSL occurrence (%) for 20c vs 21c (all sources pooled)
 C) Total casts per year (context / sampling)  [can be changed to counts of SSL if you prefer]

Bottom row (SSL casts only):
 D) Boxplot zSG by season, split 20c vs 21c
 E) Boxplot SG_core by season, split 20c vs 21c
 F) Boxplot TG_core by season, split 20c vs 21c
    (and DG_core as the last panel if you switch it in VARS list)

NOTE: By default, bottom row includes 3 vars (zSG, SG, TG).
      If you want all 4 parameters in the bottom row, either:
        - switch to a 2x4 layout, OR
        - replace the top-right panel (C) with DG boxplot.
      Here, because you asked "boxplots for all 4 parameters", I implement:
        C) DG boxplot  (and drop total casts per year).
      If you want total casts instead, flip USE_PANEL_C_FOR_DG=False.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
BASIN_LABEL = "DJ3"                      # used in title only

CAST_SUMMARY = Path(
    f"/Users/elena/Projects/IRB/MedClim/Plots/Adriatic/SSL/OUT_SSL_{BASIN_LABEL}_DUALMODE/cast_summary.csv")

START_YEAR = 1980
END_YEAR = None           # None -> max in data
MIN_CASTS_FOR_PCT = 1     # denominator threshold
SHOW_PER_SOURCE_LINES = False
KEEP_SOURCES = None       # e.g. ["ARGO","SDN","ADR_CTD_HR","ADR_IOR","ADR_RB"], or None=all

SEASONS_ORDER = ["JFM", "AMJ", "JAS", "OND"]
PERIODS_ORDER = ["20c", "21c"]

# Use panel C for DG boxplot (True) or for total casts per year (False)
USE_PANEL_C_FOR_DG = True

OUTDIR = CAST_SUMMARY.parent
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFIG = OUTDIR / f"FIG_SSL_{BASIN_LABEL}_6panels.png"
DPI = 300

# =========================
# STYLE (big labels)
# =========================
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

# =========================
# LOAD + BASIC CLEAN
# =========================
if not CAST_SUMMARY.exists():
    raise FileNotFoundError(f"Missing: {CAST_SUMMARY}")

df = pd.read_csv(CAST_SUMMARY, low_memory=False)

for col in ["source", "has_ssl"]:
    if col not in df.columns:
        raise ValueError(f"cast_summary.csv missing required column: {col}")

df["has_ssl"] = pd.to_numeric(df["has_ssl"], errors="coerce").fillna(0).astype(int)

# Year handling: prefer 'year' if present, else parse from time
if "year" in df.columns:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
else:
    if "time" not in df.columns:
        raise ValueError("No 'year' column and no 'time' column to parse year.")
    t = pd.to_datetime(df["time"], errors="coerce")
    df["year"] = t.dt.year

df = df[np.isfinite(df["year"])].copy()
df["year"] = df["year"].astype(int)

# Optional source filter
if KEEP_SOURCES is not None:
    df = df[df["source"].isin(KEEP_SOURCES)].copy()

# Season/period columns should exist from your pipeline; if not, derive period from year
if "season" not in df.columns:
    raise ValueError("cast_summary.csv missing 'season' column (expected).")
if "period" not in df.columns:
    df["period"] = np.where(df["year"] < 2000, "20c", "21c")

# Filter years
df = df[df["year"] >= START_YEAR].copy()
if df.empty:
    raise ValueError("No data after START_YEAR filter. Check CAST_SUMMARY and START_YEAR.")

if END_YEAR is None:
    END_YEAR = int(df["year"].max())

years = np.arange(START_YEAR, END_YEAR + 1, dtype=int)

# =========================
# HELPERS
# =========================
def safe_mean_pct(sub: pd.DataFrame) -> float:
    if len(sub) < MIN_CASTS_FOR_PCT:
        return np.nan
    return 100.0 * float(sub["has_ssl"].mean())

def grouped_year_table(d: pd.DataFrame) -> pd.DataFrame:
    """Return table with year, n_total, n_ssl, pct_ssl for a subset dataframe."""
    tot = d.groupby("year").size().rename("n_total")
    ssl = d[d["has_ssl"] == 1].groupby("year").size().rename("n_ssl")
    out = pd.concat([tot, ssl], axis=1).fillna(0)
    out["n_total"] = out["n_total"].astype(int)
    out["n_ssl"] = out["n_ssl"].astype(int)
    out["pct_ssl"] = np.where(out["n_total"] >= MIN_CASTS_FOR_PCT, 100.0 * out["n_ssl"] / out["n_total"], np.nan)
    out = out.reindex(years).fillna({"n_total": 0, "n_ssl": 0})
    # pct_ssl stays NaN when n_total < MIN_CASTS_FOR_PCT
    out["pct_ssl"] = np.where(out["n_total"] >= MIN_CASTS_FOR_PCT, out["pct_ssl"], np.nan)
    out = out.reset_index()
    return out

def seasonal_pct_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in SEASONS_ORDER:
        for period in PERIODS_ORDER:
            sub = d[(d["season"] == season) & (d["period"] == period)]
            rows.append({
                "season": season,
                "period": period,
                "pct": safe_mean_pct(sub),
                "n_total": int(len(sub)),
                "n_ssl": int(sub["has_ssl"].sum()),
            })
    return pd.DataFrame(rows)

def make_season_period_box_data(d_ssl: pd.DataFrame, var: str):
    """
    Returns list_of_arrays, labels, positions for boxplot:
    for each season: two boxes (20c, 21c)
    """
    data = []
    labels = []
    positions = []
    pos = 1
    gap = 1.3
    within = 0.35
    for season in SEASONS_ORDER:
        # 20c box
        v20 = d_ssl[(d_ssl["season"] == season) & (d_ssl["period"] == "20c")][var].dropna().values
        v21 = d_ssl[(d_ssl["season"] == season) & (d_ssl["period"] == "21c")][var].dropna().values
        data.extend([v20, v21])
        positions.extend([pos - within, pos + within])
        labels.append(season)
        pos += gap
    return data, labels, positions

# Variables for boxplots (must exist in cast_summary)
# If you used different column names, adjust here.
VAR_MAP = {
    "z_core": ("zSG (m)", "zSG"),
    "SG_core": ("SG (PSU m$^{-1}$)", "SG"),
    "TG_core": ("TG (°C m$^{-1}$)", "TG"),
    "DG_core": ("DG (kg m$^{-4}$)", "DG"),
}

for v in VAR_MAP.keys():
    if v not in df.columns:
        # allow missing TG/DG in some sources, but column should exist overall
        raise ValueError(f"cast_summary.csv missing expected column: {v}")

# SSL-only subset for boxplots
df_ssl = df[df["has_ssl"] == 1].copy()

# =========================
# PRECOMPUTE TABLES
# =========================
tab_all_year = grouped_year_table(df)

# Optional per-source year tables (for faint lines in panel A)
src_tables = {}
if SHOW_PER_SOURCE_LINES:
    for src in sorted(df["source"].unique()):
        src_tables[src] = grouped_year_table(df[df["source"] == src])

tab_season = seasonal_pct_table(df)

# =========================
# FIGURE LAYOUT (2x3)
# =========================
fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)

axA = axes[0, 0]  # % per year
axB = axes[0, 1]  # seasonal % 20c vs 21c
axC = axes[0, 2]  # either DG boxplot or total casts per year

axD = axes[1, 0]  # zSG box
axE = axes[1, 1]  # SG box
axF = axes[1, 2]  # TG box (and DG is in axC if enabled)

# --- Panel A: % per year (overall + optional per source)
axA.plot(tab_all_year["year"], tab_all_year["pct_ssl"], linewidth=2.5, label="All sources", color='mediumorchid')
if SHOW_PER_SOURCE_LINES:
    for src, tsrc in src_tables.items():
        # faint lines to not clutter
        axA.plot(tsrc["year"], tsrc["pct_ssl"], linewidth=1.0, alpha=0.35, label=src)

axA.set_title("A) SSL occurrence per year (%)")
axA.set_xlabel("Year")
axA.set_ylabel("SSL occurrence (%)")
xt = years[::5] if len(years) > 5 else years
axA.set_xticks(xt)
axA.set_xticklabels([str(y) for y in xt], rotation=45, ha="right")
axA.set_ylim(bottom=0)

# Legend: only show "All sources" + maybe 1 line saying "per source (faint)" to avoid huge legend
# If you really want full legend, comment the next block and uncomment axA.legend()
if SHOW_PER_SOURCE_LINES:
    axA.legend(["All sources", "Per-source lines (faint)"], loc="upper left", frameon=True)
else:
    axA.legend(loc="upper left", frameon=True)

# --- Panel B: seasonal % 20c vs 21c (pooled)
x = np.arange(len(SEASONS_ORDER))
w = 0.38
pct20 = [float(tab_season[(tab_season.season == s) & (tab_season.period == "20c")]["pct"].values[0]) for s in SEASONS_ORDER]
pct21 = [float(tab_season[(tab_season.season == s) & (tab_season.period == "21c")]["pct"].values[0]) for s in SEASONS_ORDER]

axB.bar(x - w/2, pct20, width=w, label="20c", color='mediumorchid')
axB.bar(x + w/2, pct21, width=w, label="21c", color='darkorange')
axB.set_title("B) Seasonal SSL occurrence (%) — 20c vs 21c")
axB.set_xlabel("Season")
axB.set_ylabel("SSL occurrence (%)")
axB.set_xticks(x)
axB.set_xticklabels(SEASONS_ORDER)
axB.set_ylim(bottom=0)
axB.legend(loc="upper left", frameon=True)

# Add sample sizes under bars (n_ssl/n_total)
for i, s in enumerate(SEASONS_ORDER):
    r20 = tab_season[(tab_season.season == s) & (tab_season.period == "20c")].iloc[0]
    r21 = tab_season[(tab_season.season == s) & (tab_season.period == "21c")].iloc[0]
    txt20 = f"{int(r20.n_ssl)}/{int(r20.n_total)}"
    txt21 = f"{int(r21.n_ssl)}/{int(r21.n_total)}"
    axB.text(i - w/2, 0.5, txt20, rotation=90, va="bottom", ha="center", fontsize=10)
    axB.text(i + w/2, 0.5, txt21, rotation=90, va="bottom", ha="center", fontsize=10)

# --- Panel C: either DG boxplot OR total casts per year
if USE_PANEL_C_FOR_DG:
    # DG boxplot (SSL casts only)
    data, labels, positions = make_season_period_box_data(df_ssl, "DG_core")
    bp = axC.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    axC.set_title("C) DG — by season, 20c vs 21c")
    axC.set_ylabel(VAR_MAP["DG_core"][0])
    # season ticks at midpoints
    mids = [np.mean([positions[2*i], positions[2*i+1]]) for i in range(len(SEASONS_ORDER))]
    axC.set_xticks(mids)
    axC.set_xticklabels(labels)
    # light separators between seasons
    for i in range(1, len(SEASONS_ORDER)):
        axC.axvline(mids[i] - 0.65, color="0.85", linewidth=1.0)
    # simple legend by coloring boxes alternately
    for j, box in enumerate(bp["boxes"]):
        # even index -> 20c, odd -> 21c
        box.set_alpha(0.8)
        # do not hardcode colors; matplotlib default cycles are fine for paper, but boxplot needs facecolor:
        # keep grayscale-ish to avoid forced colors
        if j % 2 == 0:
            box.set_facecolor("0.85")
        else:
            box.set_facecolor("0.65")
    axC.text(0.02, 0.98, "Box color: 20c (light) / 21c (dark)", transform=axC.transAxes,
             ha="left", va="top", fontsize=11)
else:
    axC.plot(tab_all_year["year"], tab_all_year["n_total"], linewidth=2.0)
    axC.set_title("C) Total casts per year (sampling)")
    axC.set_xlabel("Year")
    axC.set_ylabel("Number of casts")
    axC.set_xticks(xt)
    axC.set_xticklabels([str(y) for y in xt], rotation=45, ha="right")
    axC.set_ylim(bottom=0)

# --- Bottom row: zSG, SG, TG boxplots (SSL casts only)
def draw_box(ax, var, panel_title):
    data, labels, positions = make_season_period_box_data(df_ssl, var)
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    ax.set_title(panel_title)
    ax.set_ylabel(VAR_MAP[var][0])

    mids = [np.mean([positions[2*i], positions[2*i+1]]) for i in range(len(SEASONS_ORDER))]
    ax.set_xticks(mids)
    ax.set_xticklabels(labels)

    for i in range(1, len(SEASONS_ORDER)):
        ax.axvline(mids[i] - 0.65, color="0.85", linewidth=1.0)

    # 20c vs 21c shading
    for j, box in enumerate(bp["boxes"]):
        box.set_alpha(0.8)
        if j % 2 == 0:
            box.set_facecolor("0.85")  # 20c
        else:
            box.set_facecolor("0.65")  # 21c

    # Add small legend once per axis
    ax.text(0.02, 0.02, "20c light / 21c dark", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11)

draw_box(axD, "z_core", "D) zSG — by season, 20c vs 21c")
draw_box(axE, "SG_core", "E) SG — by season, 20c vs 21c")
draw_box(axF, "TG_core", "F) TG — by season, 20c vs 21c")

# Overall figure title (kept high but not cramped)

# Save
fig.savefig(OUTFIG, dpi=DPI, bbox_inches="tight")
plt.close(fig)

print("Wrote:", OUTFIG.resolve())
print("Done.")