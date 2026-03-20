#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# INPUT / OUTPUT
# -------------------------
# Prefer the cleaned full-schema file (keeps instrument column reliably)
FILE = Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.clean.csv")

OUTDIR = Path("/Users/elena/Projects/IRB/MedClim/Plots/Seadatanet") / "Stats"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTDIR / "casts_per_year_platform.csv"
OUT_PNG = OUTDIR / "casts_per_year_platform.png"

# -------------------------
# COLUMNS
# -------------------------
DATE_COL = "yyyy-mm-ddThh:mm:ss.sss"
DEPTH_COL = "Depth [m]"  # used to keep only profile rows
INST_COL = "Instrument / gear type:METAVAR:INDEXED_TEXT"

# Cast identity: use LOCAL_CDI_ID when possible, else fallback
CAST_COL = "LOCAL_CDI_ID"
CRUISE_COL = "Cruise"
STATION_COL = "Station:METAVAR:INDEXED_TEXT"

CHUNKSIZE = 500_000

# -------------------------
# PLATFORM CLASSIFICATION
# -------------------------
# Edit these as needed to match your export vocabulary
CTD_KEYS = ("ctd",)
BOTTLE_KEYS = ("bottle", "discrete water sampler", "water sampler", "niskin")
XBT_KEYS = ("xbt", "mbt", "bathythermograph", "expendable bathythermograph", "thermograph")
TSG_KEYS = ("thermosalinograph", "thermosalinographs")  # (very surfacey; keep separate if you want)

def classify_platform(instr: str) -> str:
    s = (instr or "").strip().lower()
    if not s or s in ("nan", "none"):
        return "Unknown"
    # You can decide whether to treat TSG as its own group or fold it into XBT-like
    if any(k in s for k in CTD_KEYS):
        return "CTD"
    if any(k in s for k in BOTTLE_KEYS):
        return "Bottle"
    if any(k in s for k in XBT_KEYS):
        return "MBT/XBT"
    if any(k in s for k in TSG_KEYS):
        return "TSG"  # optional group
    return "Other"

def build_cast_id(df: pd.DataFrame) -> pd.Series:
    cid = df.get(CAST_COL, pd.Series([""] * len(df), index=df.index)).astype(str).str.strip()
    good = cid.ne("") & cid.ne("nan") & cid.ne("None")

    t = pd.to_datetime(df[DATE_COL], errors="coerce")
    date = t.dt.strftime("%Y-%m-%d").fillna("")

    fb = (
        df.get(CRUISE_COL, "").astype(str).fillna("") + "|" +
        df.get(STATION_COL, "").astype(str).fillna("") + "|" +
        date
    )
    return cid.where(good, fb)

def main():
    header = pd.read_csv(FILE, nrows=0).columns.tolist()
    have_inst = (INST_COL in header)

    usecols = [DATE_COL, DEPTH_COL, CAST_COL, CRUISE_COL, STATION_COL]
    if have_inst:
        usecols.append(INST_COL)

    print("Instrument column present:", have_inst)
    print("Reading columns:", usecols)

    # We want casts per year (unique cast ids), so we store per-year sets of casts.
    # For platform breakdown, store per (year, platform) sets.
    casts_by_year = defaultdict(set)
    casts_by_year_platform = defaultdict(set)

    total_rows = 0
    for k, ch in enumerate(pd.read_csv(FILE, usecols=usecols, chunksize=CHUNKSIZE, low_memory=False, dtype=str)):
        total_rows += len(ch)

        # keep only rows with a Depth value => true profile samples
        ch = ch.dropna(subset=[DATE_COL, DEPTH_COL])
        if ch.empty:
            continue

        dt = pd.to_datetime(ch[DATE_COL], errors="coerce")
        ch = ch.loc[dt.notna()].copy()
        if ch.empty:
            continue

        year = dt.loc[ch.index].dt.year.astype(int)
        ch["year"] = year

        ch["cast"] = build_cast_id(ch)

        if have_inst:
            ch["platform"] = ch[INST_COL].astype(str).map(classify_platform)
        else:
            ch["platform"] = "All"

        # We need unique (year, cast) and (year, platform, cast) across chunks
        # Reduce inside chunk
        yc = ch[["year", "cast"]].drop_duplicates()
        for y, c in yc.itertuples(index=False):
            casts_by_year[int(y)].add(c)

        ypc = ch[["year", "platform", "cast"]].drop_duplicates()
        for y, p, c in ypc.itertuples(index=False):
            casts_by_year_platform[(int(y), str(p))].add(c)

        if (k % 20) == 0:
            print(f"chunk {k:5d} rows_read={total_rows:,} years_seen={len(casts_by_year):,}")

    # Build summary table
    years = sorted(casts_by_year.keys())
    total = np.array([len(casts_by_year[y]) for y in years], dtype=int)

    # Platforms to plot (order matters)
    platforms = ["MBT/XBT", "Bottle", "CTD"]
    if have_inst:
        # add TSG if present at all
        if any(p == "TSG" for (_, p) in casts_by_year_platform.keys()):
            platforms.insert(0, "TSG")
        # also keep Other/Unknown if you want
        # platforms += ["Other", "Unknown"]

    data = {"year": years, "Total": total}
    for p in platforms:
        data[p] = [len(casts_by_year_platform.get((y, p), set())) for y in years]

    out = pd.DataFrame(data)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}")

    # Plot: stacked bars if we have platforms, else single bars
    plt.figure(figsize=(14, 4.5))
    x = np.arange(len(years))

    if have_inst:
        bottom = np.zeros_like(x, dtype=float)
        for p in platforms:
            vals = out[p].to_numpy(float)
            plt.bar(years, vals, bottom=bottom, label=p)
            bottom += vals
        plt.legend(loc="upper right", frameon=True)
        plt.ylabel("Number of casts (unique profiles)")
        plt.title("SeaDataNet casts per year (by platform class; heuristic mapping)")
    else:
        plt.bar(years, out["Total"].to_numpy(float))
        plt.ylabel("Number of casts (unique profiles)")
        plt.title("SeaDataNet casts per year")

    plt.xlabel("Year")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print(f"Wrote: {OUT_PNG}")

if __name__ == "__main__":
    main()
