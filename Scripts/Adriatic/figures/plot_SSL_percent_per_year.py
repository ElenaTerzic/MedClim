#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_ssl_per_year.py

Read a cast_summary.csv produced by SSL_profiles_* and plot:
  1) SSL counts per year by source
  2) SSL occurrence (%) per year by source  [= SSL casts / total casts * 100]

Key robustness:
- If column 'year' exists -> use it (DO NOT re-parse time)
- Else parse from 'time'
- Prints diagnostics that immediately reveal if you're reading the wrong file
  (e.g., ARGO missing, or ARGO only in one year, etc.)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
TARGET_BASIN = "DJ3"

CAST_SUMMARY = Path(
    f"/Users/elena/Projects/IRB/MedClim/Plots/Adriatic/SSL/OUT_SSL_{TARGET_BASIN}_DUALMODE/cast_summary.csv"
)
START_YEAR = 1980
END_YEAR = None  # None = max year in data
MIN_TOTAL_CASTS_FOR_PCT = 1  # keep % only when denominator >= this
FIG_DPI = 150

# If you want to limit which sources appear (optional), set list; else None
KEEP_SOURCES = None  # e.g. ["ARGO","SDN","ADR_CTD_HR","ADR_IOR","ADR_RB"]

# =========================
# LOAD
# =========================
if not CAST_SUMMARY.exists():
    raise FileNotFoundError(f"Missing cast summary: {CAST_SUMMARY}")

df = pd.read_csv(CAST_SUMMARY, low_memory=False)

required = {"source", "has_ssl"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"cast_summary.csv missing required columns: {sorted(missing)}")

# Make sure has_ssl is numeric 0/1
df["has_ssl"] = pd.to_numeric(df["has_ssl"], errors="coerce").fillna(0).astype(int)

# Prefer 'year' column if present; else parse from time
if "year" in df.columns:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
else:
    if "time" not in df.columns:
        raise ValueError("No 'year' column and no 'time' column available to parse year.")
    t = pd.to_datetime(df["time"], errors="coerce")
    df["year"] = t.dt.year

df = df[np.isfinite(df["year"])].copy()
df["year"] = df["year"].astype(int)

# Optional: filter sources
if KEEP_SOURCES is not None:
    df = df[df["source"].isin(KEEP_SOURCES)].copy()

# Filter start year
df = df[df["year"] >= START_YEAR].copy()

if df.empty:
    raise ValueError("No rows left after filtering by year/source. Check CAST_SUMMARY path and filters.")

# Determine end year
if END_YEAR is None:
    END_YEAR = int(df["year"].max())

years = np.arange(START_YEAR, END_YEAR + 1, dtype=int)

# =========================
# DIAGNOSTICS (IMPORTANT)
# =========================
print("\n=== DIAGNOSTICS ===")
print("Input:", CAST_SUMMARY.resolve())
print("Rows:", len(df))
print("Sources:", df["source"].unique())

print("\nTotal casts by source:")
print(df.groupby("source").size().sort_values(ascending=False))

print("\nSSL casts by source:")
print(df[df["has_ssl"] == 1].groupby("source").size().sort_values(ascending=False))

print("\nYear range by source:")
print(df.groupby("source")["year"].agg(["min", "max", "nunique"]).sort_values("min"))

# This table immediately tells you if ARGO is only in one year (bad) or many (good)
print("\nSSL counts per year by source (top 20 rows):")
ssl_year_src = (
    df[df["has_ssl"] == 1]
    .groupby(["source", "year"])
    .size()
    .reset_index(name="n_ssl")
    .sort_values(["source", "year"])
)
print(ssl_year_src.head(20).to_string(index=False))

# =========================
# AGGREGATE (totals + ssl)
# =========================
tot = (
    df.groupby(["source", "year"])
    .size()
    .rename("n_total")
    .reset_index()
)

ssl = (
    df[df["has_ssl"] == 1]
    .groupby(["source", "year"])
    .size()
    .rename("n_ssl")
    .reset_index()
)

merged = tot.merge(ssl, on=["source", "year"], how="left")
merged["n_ssl"] = merged["n_ssl"].fillna(0).astype(int)
merged["pct_ssl"] = np.where(
    merged["n_total"] >= MIN_TOTAL_CASTS_FOR_PCT,
    100.0 * merged["n_ssl"] / merged["n_total"],
    np.nan
)

sources = sorted(merged["source"].unique())

# A consistent bar layout: small offsets for each source within a year
nS = len(sources)
bar_w = 0.80 / max(nS, 1)

# =========================
# PLOT: COUNTS
# =========================
fig = plt.figure(figsize=(16, 4.8))
ax = plt.gca()

for i, src in enumerate(sources):
    sub = merged[merged["source"] == src].set_index("year").reindex(years)
    y = sub["n_ssl"].fillna(0).to_numpy()
    x = years + (i - (nS - 1) / 2.0) * bar_w
    ax.bar(x, y, width=bar_w, label=src)

ax.set_title("SSL counts per year by source")
ax.set_xlabel("Year")
ax.set_ylabel("Number of SSL casts")

# ticks every 5 years
xt = years[::5] if len(years) > 5 else years
ax.set_xticks(xt)
ax.set_xticklabels([str(y) for y in xt], rotation=45, ha="right")

ax.legend(loc="upper left", frameon=True)
ax.grid(False)
plt.tight_layout()

out_counts = CAST_SUMMARY.parent / f"ssl_counts_per_year_by_source_from_{START_YEAR}.png"
plt.savefig(out_counts, dpi=FIG_DPI)
plt.close(fig)

# =========================
# PLOT: PERCENT
# =========================
fig = plt.figure(figsize=(16, 4.8))
ax = plt.gca()

for i, src in enumerate(sources):
    sub = merged[merged["source"] == src].set_index("year").reindex(years)
    y = sub["pct_ssl"].to_numpy()
    x = years + (i - (nS - 1) / 2.0) * bar_w
    ax.bar(x, y, width=bar_w, label=src)

ax.set_title("SSL occurrence (%) per year by source")
ax.set_xlabel("Year")
ax.set_ylabel("SSL occurrence (%)")

xt = years[::5] if len(years) > 5 else years
ax.set_xticks(xt)
ax.set_xticklabels([str(y) for y in xt], rotation=45, ha="right")

ax.legend(loc="upper left", frameon=True)
ax.grid(False)
plt.tight_layout()

out_pct = CAST_SUMMARY.parent / f"ssl_pct_per_year_by_source_from_{START_YEAR}.png"
plt.savefig(out_pct, dpi=FIG_DPI)
plt.close(fig)

print("\nWrote:")
print(" ", out_counts)
print(" ", out_pct)
print("Done.")