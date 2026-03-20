#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
FILE = Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only.csv")
OUTDIR = Path("/Users/elena/Projects/IRB/MedClim/Plots/Seadatanet") / "Stats"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTDIR / "depth_coverage_by_cast.csv"
OUT_PNG = OUTDIR / "depth_coverage_by_cast.png"

# ---------------- Columns ----------------
DATE_COL   = "yyyy-mm-ddThh:mm:ss.sss"
DEPTH_COL  = "Depth [m]"
CAST_COL   = "LOCAL_CDI_ID"
CRUISE_COL = "Cruise"
STATION_COL= "Station:METAVAR:INDEXED_TEXT"

CHUNKSIZE = 500_000

# Depth bins like the report figure
BINS = [0, 30, 150, 300, 600, 1000, 2000, 3000, 4000, 5500]
BIN_LABELS = [f"{BINS[i]}-{BINS[i+1]}" for i in range(len(BINS)-1)]

def build_cast_id(df: pd.DataFrame) -> pd.Series:
    """
    Prefer LOCAL_CDI_ID when non-empty; otherwise fallback to Cruise|Station|date.
    """
    cid = df.get(CAST_COL)
    if cid is None:
        cid = pd.Series([""] * len(df), index=df.index)

    cid = cid.astype(str).str.strip()
    has = cid.ne("") & cid.ne("nan") & cid.ne("None")

    # fallback cast key
    t = pd.to_datetime(df[DATE_COL], errors="coerce")
    date = t.dt.strftime("%Y-%m-%d").fillna("")

    fb = (
        df[CRUISE_COL].astype(str).fillna("") + "|" +
        df[STATION_COL].astype(str).fillna("") + "|" +
        date
    )

    out = cid.where(has, fb)
    return out

def main():
    usecols = [DATE_COL, DEPTH_COL, CAST_COL, CRUISE_COL, STATION_COL]
    # Track maximum depth per cast (streaming)
    zmax = {}

    total_rows = 0
    for k, ch in enumerate(pd.read_csv(FILE, usecols=usecols, chunksize=CHUNKSIZE, low_memory=False, dtype=str)):
        total_rows += len(ch)

        # parse
        ch = ch.dropna(subset=[DEPTH_COL, DATE_COL])
        if ch.empty:
            continue

        ch[DEPTH_COL] = pd.to_numeric(ch[DEPTH_COL], errors="coerce")
        ch = ch.dropna(subset=[DEPTH_COL])
        if ch.empty:
            continue

        ch["_cast"] = build_cast_id(ch)

        # reduce to per-cast max depth in this chunk
        g = ch.groupby("_cast", sort=False)[DEPTH_COL].max()

        # merge into global dict
        for cast, zm in g.items():
            if not np.isfinite(zm):
                continue
            prev = zmax.get(cast)
            if prev is None or zm > prev:
                zmax[cast] = float(zm)

        if (k % 20) == 0:
            print(f"chunk {k:5d} rows_read={total_rows:,} casts_seen={len(zmax):,}")

    print(f"\nDone scanning.")
    print(f"Total casts: {len(zmax):,}")

    zvals = np.array(list(zmax.values()), float)

    # Count casts by zmax-bin (casts reaching each bin top)
    # Here we count by which interval the *maximum depth* falls in.
    counts, _ = np.histogram(zvals, bins=BINS)
    summary = pd.DataFrame({"depth_bin_m": BIN_LABELS, "n_casts": counts})
    summary.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(summary["depth_bin_m"], summary["n_casts"])
    plt.ylabel("Number of casts")
    plt.xlabel("Maximum cast depth bin (m)")
    plt.title("SeaDataNet casts by maximum depth (profiles_only)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print(f"Wrote: {OUT_PNG}")

if __name__ == "__main__":
    main()
