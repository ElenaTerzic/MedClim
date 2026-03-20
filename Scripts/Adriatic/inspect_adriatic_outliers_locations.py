#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect Adriatic 'out' casts (BASIN_3ADR == 'out'):

Outputs:
- adriatic_out_casts_summary.csv  (one row per OUT cast)
- adriatic_dj_casts_summary.csv   (one row per DJ cast)
- adriatic_out_casts_map.png      (scatter plot: DJ casts (grey) + OUT casts (red))
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = Path("/Users/elena/Projects/IRB/MedClim/Diagnostics/AdriaticOut")
OUTDIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("CTD_HR_2019_2023_MFSD",
     Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/CTD_HR_2019_2023_MFSD_long.csv")),
    ("IOR_Adriatic_database_1987_2019",
     Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/IOR_Adriatic_database_1987_2019_long.csv")),
]

CHUNKSIZE = 400_000
DJ_SET = {"DJ1", "DJ2", "DJ3"}

# candidate column names
TIME_CANDS   = ["TIME", "time", "DATE_TIME", "datetime"]
LAT_CANDS    = ["LATITUDE", "Latitude", "LAT", "lat", "Latitude [degrees_north]"]
LON_CANDS    = ["LONGITUDE", "Longitude", "LON", "lon", "Longitude [degrees_east]"]
STATION_CANDS= ["STATION", "Station", "station"]
BASIN_CANDS  = ["BASIN_3ADR", "basin", "BASIN"]
DEPTH_CANDS  = ["DEPTH_M", "Depth [m]", "DEPTH", "depth"]

def pick_col(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None

def norm(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s

def cast_summary(df: pd.DataFrame, source_name: str,
                 time_col: str, lat_col: str, lon_col: str,
                 station_col: str, basin_col: str, depth_col: str) -> pd.DataFrame:
    # numeric columns
    df = df.copy()
    df["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["_z"]   = pd.to_numeric(df[depth_col], errors="coerce")

    # cast id = station + time string
    df["_cast"] = df[station_col].astype(str) + "|" + df[time_col].astype(str)

    g = df.groupby("_cast", as_index=False).agg(
        station=(station_col, "first"),
        time=(time_col, "first"),
        basin_raw=(basin_col, "first"),
        lat_med=("_lat", "median"),
        lon_med=("_lon", "median"),
        zmin=("_z", "min"),
        zmax=("_z", "max"),
        nrows=("_cast", "size"),
    )
    g["source"] = source_name
    return g

def coord_check(df, label):
    bad = (
        df["lat_med"].isna() | df["lon_med"].isna() |
        (df["lat_med"].abs() > 90) | (df["lon_med"].abs() > 180)
    )
    print(f"\nCoord check [{label}]:")
    print("  missing/invalid coords casts:", int(bad.sum()))
    print(f"  lat range: {df['lat_med'].min():.4f} .. {df['lat_med'].max():.4f}")
    print(f"  lon range: {df['lon_med'].min():.4f} .. {df['lon_med'].max():.4f}")

def main():
    out_casts_all = []
    dj_casts_all = []

    for name, path in SOURCES:
        if not path.exists():
            print(f"[MISSING] {name}: {path}")
            continue

        # detect columns from header
        head = pd.read_csv(path, nrows=5, dtype=str, low_memory=False)
        cols = list(head.columns)

        time_col    = pick_col(cols, TIME_CANDS)
        lat_col     = pick_col(cols, LAT_CANDS)
        lon_col     = pick_col(cols, LON_CANDS)
        station_col = pick_col(cols, STATION_CANDS)
        basin_col   = pick_col(cols, BASIN_CANDS)
        depth_col   = pick_col(cols, DEPTH_CANDS)

        missing = [k for k, v in {
            "time_col": time_col, "lat_col": lat_col, "lon_col": lon_col,
            "station_col": station_col, "basin_col": basin_col, "depth_col": depth_col
        }.items() if v is None]

        print(f"\n[{name}] reading {path.name}")
        if missing:
            print("  Missing required columns:", missing)
            print("  Columns in file (first 40):", cols[:40])
            continue

        usecols = [time_col, lat_col, lon_col, station_col, basin_col, depth_col]
        chunks = pd.read_csv(path, chunksize=CHUNKSIZE, dtype=str, low_memory=False,
                             usecols=lambda c: c in set(usecols))

        out_parts = []
        dj_parts = []

        for chunk in chunks:
            basin = chunk[basin_col].map(norm).str.lower()

            m_out = basin.eq("out")
            m_dj  = basin.isin({x.lower() for x in DJ_SET})

            if m_out.any():
                out_parts.append(chunk.loc[m_out].copy())
            if m_dj.any():
                dj_parts.append(chunk.loc[m_dj].copy())

        if out_parts:
            out_df = pd.concat(out_parts, ignore_index=True)
            out_casts_all.append(
                cast_summary(out_df, name, time_col, lat_col, lon_col, station_col, basin_col, depth_col)
            )
            print(f"  OUT rows: {len(out_df)}")
        else:
            print("  OUT rows: 0")

        if dj_parts:
            dj_df = pd.concat(dj_parts, ignore_index=True)
            dj_casts_all.append(
                cast_summary(dj_df, name, time_col, lat_col, lon_col, station_col, basin_col, depth_col)
            )
            print(f"  DJ rows: {len(dj_df)}")
        else:
            print("  DJ rows: 0")

    if not out_casts_all:
        print("\nNo OUT casts found in these sources.")
        return

    out_casts = pd.concat(out_casts_all, ignore_index=True)
    out_csv = OUTDIR / "adriatic_out_casts_summary.csv"
    out_casts.to_csv(out_csv, index=False)
    print("\nWrote:", out_csv, "casts:", len(out_casts))

    if dj_casts_all:
        dj_casts = pd.concat(dj_casts_all, ignore_index=True)
        dj_csv = OUTDIR / "adriatic_dj_casts_summary.csv"
        dj_casts.to_csv(dj_csv, index=False)
        print("Wrote:", dj_csv, "casts:", len(dj_casts))
    else:
        dj_casts = None

    coord_check(out_casts, "OUT")
    if dj_casts is not None:
        coord_check(dj_casts, "DJ")

    # plot map
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    if dj_casts is not None:
        ax.scatter(dj_casts["lon_med"], dj_casts["lat_med"], s=4, alpha=0.25, label="DJ casts (all)")
    ax.scatter(out_casts["lon_med"], out_casts["lat_med"], s=18, alpha=0.9, label="OUT casts")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Adriatic casts: DJ vs OUT (BASIN_3ADR=='out')")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    out_png = OUTDIR / "adriatic_out_casts_map.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_png)

    print("\nTop OUT stations (by casts):")
    print(out_casts["station"].astype(str).value_counts().head(20).to_string())

    print("\nDone.")

if __name__ == "__main__":
    main()
