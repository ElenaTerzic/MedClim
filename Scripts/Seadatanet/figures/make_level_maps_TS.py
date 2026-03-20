#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# INPUT / OUTPUT
# -----------------------------
INFILE = Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.clean.csv")
OUTDIR = Path("/Users/elena/Projects/IRB/MedClim/Plots/Seadatanet") / "Plots_Level_Maps"
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# COLUMNS (from your ODV export)
# -----------------------------
DATE_COL  = "yyyy-mm-ddThh:mm:ss.sss"
LON_COL   = "Longitude [degrees_east]"
LAT_COL   = "Latitude [degrees_north]"
CAST_COL  = "LOCAL_CDI_ID"
CRUISE_COL = "Cruise"
STATION_COL = "Station:METAVAR:INDEXED_TEXT"

DEPTH_COL = "Depth [m]"
TEMP_COL  = "ITS-90 water temperature [degrees C]"
SAL_COL   = "Water body salinity [per mille]"

# QC columns in ODV export (usually aligned with variables):
# Depth [m], QV:SEADATANET, Temp, QV:SEADATANET.1, Sal, QV:SEADATANET.2
TEMP_QC_COL = "QV:SEADATANET.1"
SAL_QC_COL  = "QV:SEADATANET.2"

# -----------------------------
# MAP LEVELS & TOLERANCES
# -----------------------------
LEVELS = [15, 30, 150, 300, 600, 1000, 2000, 3000]

# tolerance (meters) for "nearest sample to depth"
# (tweak if you want stricter/looser matching)
TOL = {
    15:  2.5,
    30:  3.0,
    150: 10.0,
    300: 15.0,
    600: 25.0,
    1000: 30.0,
    2000: 60.0,
    3000: 100.0,
}

# QC filter (report says QF=1,2)
GOOD_QF = {"1", "2", 1, 2}

CHUNKSIZE = 500_000

# -----------------------------
# CAST ID (robust)
# -----------------------------
def build_cast_id(df: pd.DataFrame) -> pd.Series:
    """
    Use LOCAL_CDI_ID if present/non-empty, else fallback to Cruise|Station|YYYY-MM-DD.
    """
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

# -----------------------------
# CORE SELECTOR (streaming)
# -----------------------------
def update_best(best_dict, cast, absdiff, lon, lat, val):
    """
    Keep best (smallest absdiff) point per cast for a given level.
    best_dict[cast] = (absdiff, lon, lat, val)
    """
    prev = best_dict.get(cast)
    if prev is None or absdiff < prev[0]:
        best_dict[cast] = (absdiff, lon, lat, val)

def collect_points_for_var(var_col: str, qc_col: str | None, out_prefix: str):
    """
    Returns dict: level -> best_points_by_cast
    best_points_by_cast: cast -> (absdiff, lon, lat, value)
    """
    # store per level
    best = {lev: {} for lev in LEVELS}

    # choose minimal read columns
    usecols = [DATE_COL, LON_COL, LAT_COL, DEPTH_COL, var_col, CAST_COL, CRUISE_COL, STATION_COL]
    if qc_col is not None:
        usecols.append(qc_col)

    # Some columns might not exist; handle gracefully
    header = pd.read_csv(INFILE, nrows=0).columns
    usecols = [c for c in usecols if c in header]
    qc_present = (qc_col is not None and qc_col in header)

    print(f"\nCollecting {out_prefix} points…")
    print("QC filter enabled:", qc_present)

    total = 0
    for k, ch in enumerate(pd.read_csv(INFILE, usecols=usecols, chunksize=CHUNKSIZE, low_memory=False, dtype=str)):
        total += len(ch)

        # numeric conversions
        ch = ch.dropna(subset=[LON_COL, LAT_COL, DEPTH_COL, var_col, DATE_COL])
        if ch.empty:
            continue

        # QC filter if available
        if qc_present:
            q = ch[qc_col].astype(str).str.strip()
            ch = ch[q.isin({str(x) for x in GOOD_QF})]
            if ch.empty:
                continue

        ch["lon"] = pd.to_numeric(ch[LON_COL], errors="coerce")
        ch["lat"] = pd.to_numeric(ch[LAT_COL], errors="coerce")
        ch["z"]   = pd.to_numeric(ch[DEPTH_COL], errors="coerce")
        ch["v"]   = pd.to_numeric(ch[var_col], errors="coerce")

        ch = ch.dropna(subset=["lon", "lat", "z", "v"])
        if ch.empty:
            continue

        ch["cast"] = build_cast_id(ch)

        # For each target level: filter within tolerance, pick nearest per cast within chunk, merge into global
        for lev in LEVELS:
            tol = TOL[lev]
            sub = ch[(ch["z"] >= lev - tol) & (ch["z"] <= lev + tol)]
            if sub.empty:
                continue

            sub = sub.copy()
            sub["absdz"] = (sub["z"] - lev).abs()

            # pick nearest row per cast inside chunk
            idx = sub.groupby("cast")["absdz"].idxmin()
            pick = sub.loc[idx, ["cast", "absdz", "lon", "lat", "v"]]

            # update global best
            b = best[lev]
            for row in pick.itertuples(index=False):
                update_best(b, row.cast, float(row.absdz), float(row.lon), float(row.lat), float(row.v))

        if (k % 20) == 0:
            nlev = {lev: len(best[lev]) for lev in LEVELS}
            print(f"chunk {k:5d} rows_read={total:,} points_by_level={nlev}")

    # Write per-level CSVs
    for lev in LEVELS:
        rows = []
        for cast, (absdz, lon, lat, val) in best[lev].items():
            rows.append((cast, lon, lat, val, absdz))
        df = pd.DataFrame(rows, columns=["cast", "lon", "lat", out_prefix, "absdz_m"])
        df.to_csv(OUTDIR / f"{out_prefix}_level_{lev:04d}m.csv", index=False)

    return best

# -----------------------------
# PLOTTING
# -----------------------------
def plot_8panel(best, out_prefix: str, title: str, units: str, outfile: Path):
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), constrained_layout=True)
    axes = axes.ravel()

    for i, lev in enumerate(LEVELS):
        ax = axes[i]
        d = best[lev]
        if not d:
            ax.set_title(f"{lev} m (no data)")
            ax.set_xlim(-6, 37)
            ax.set_ylim(30, 46)
            ax.grid(True, alpha=0.3)
            continue

        lon = np.array([v[1] for v in d.values()], float)
        lat = np.array([v[2] for v in d.values()], float)
        val = np.array([v[3] for v in d.values()], float)

        sc = ax.scatter(lon, lat, c=val, s=2, linewidths=0)

        ax.set_title(f"{out_prefix} @ {lev} m  (n={len(val):,})")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_xlim(-6, 37)
        ax.set_ylim(30, 46)
        ax.grid(True, alpha=0.25)

        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(units)

    fig.suptitle(title, fontsize=16)
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"Wrote: {outfile}")

def main():
    # Temperature
    bestT = collect_points_for_var(TEMP_COL, TEMP_QC_COL, out_prefix="TEMP_C")
    plot_8panel(
        bestT,
        out_prefix="TEMP_C",
        title="SeaDataNet/ODV: Temperature (QF=1,2 if available) at fixed depths (nearest sample within tolerance)",
        units="°C",
        outfile=OUTDIR / "maps_temperature_levels.png"
    )

    # Salinity
    bestS = collect_points_for_var(SAL_COL, SAL_QC_COL, out_prefix="SAL_permille")
    plot_8panel(
        bestS,
        out_prefix="SAL_permille",
        title="SeaDataNet/ODV: Salinity (QF=1,2 if available) at fixed depths (nearest sample within tolerance)",
        units="per mille",
        outfile=OUTDIR / "maps_salinity_levels.png"
    )

if __name__ == "__main__":
    main()
