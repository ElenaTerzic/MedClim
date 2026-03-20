#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reassign Adriatic casts tagged as BASIN_3ADR == 'out' into DJ1/DJ2/DJ3,
using nearest-centroid classification based on casts already tagged as
nadr/madr/sadr (or DJ1/DJ2/DJ3 if present).

Outputs (for each input file):
- <original>_fixed.csv  (adds BASIN_3ADR_FIXED column)

Notes:
- Does NOT overwrite BASIN_3ADR.
- Works even if each individual file contains only a subset of DJ1/2/3
  (centroids are computed globally across all files first).
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ------------------------
# INPUTS
# ------------------------
SOURCES = [
    Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/CTD_HR_2019_2023_MFSD_long.csv"),
    Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/RB_Northern_Adriatic_Transect_1979_2018_long.csv"),
    Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/Palagruza_Transect_1952_2018_long.csv"),
    Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/IOR_Adriatic_database_1987_2019_long.csv"),
]

CHUNKSIZE = 400_000

DJ_SET = {"DJ1", "DJ2", "DJ3"}

# This is the key: BASIN_3ADR remap → DJ
ADR3_TO_DJ = {
    "nadr": "DJ1",
    "north": "DJ1",

    "madr": "DJ2",
    "mid": "DJ2",
    "middle": "DJ2",
    "central": "DJ2",

    "sadr": "DJ3",
    "south": "DJ3",

    # allow already-mapped values too
    "dj1": "DJ1",
    "dj2": "DJ2",
    "dj3": "DJ3",
}


# ------------------------
# Column candidates (robust)
# ------------------------
TIME_CANDS    = ["TIME", "time", "DATE_TIME", "datetime"]
LAT_CANDS     = ["LATITUDE", "Latitude", "LAT", "lat"]
LON_CANDS     = ["LONGITUDE", "Longitude", "LON", "lon"]
STATION_CANDS = ["STATION", "Station", "station"]
BASIN_CANDS   = ["BASIN_3ADR", "basin", "BASIN"]


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


def remap_adr3_to_dj(val: str) -> str:
    """
    Map BASIN_3ADR values to DJ1/2/3 where possible.
    Unmapped values returned as-is (e.g. 'out').
    """
    s = norm(val).lower()
    if s in ADR3_TO_DJ:
        return ADR3_TO_DJ[s]
    return norm(val)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.deg2rad(lat1); lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2); lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def build_cast_id(df, station_col, time_col):
    # Same cast key style you’ve been using: station + "|" + time string
    return df[station_col].astype(str) + "|" + df[time_col].astype(str)


def compute_cast_positions(chunk, station_col, time_col, lat_col, lon_col):
    df = chunk.copy()
    df["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["_cast"] = build_cast_id(df, station_col, time_col)
    g = df.groupby("_cast", as_index=False).agg(
        lat_med=("_lat", "median"),
        lon_med=("_lon", "median"),
    )
    return g


def main():
    # ---------- Detect columns once per file ----------
    file_specs = []
    for path in SOURCES:
        if not path.exists():
            print("[MISSING]", path)
            continue

        head = pd.read_csv(path, nrows=5, dtype=str, low_memory=False)
        cols = list(head.columns)

        time_col    = pick_col(cols, TIME_CANDS)
        lat_col     = pick_col(cols, LAT_CANDS)
        lon_col     = pick_col(cols, LON_CANDS)
        station_col = pick_col(cols, STATION_CANDS)
        basin_col   = pick_col(cols, BASIN_CANDS)

        missing = [k for k, v in {
            "time_col": time_col,
            "lat_col": lat_col,
            "lon_col": lon_col,
            "station_col": station_col,
            "basin_col": basin_col,
        }.items() if v is None]

        if missing:
            print("\n=== Processing:", path.name, "===")
            print("  Missing required columns:", missing)
            print("  Columns (first 60):", cols[:60])
            continue

        file_specs.append((path, time_col, lat_col, lon_col, station_col, basin_col))

    if not file_specs:
        print("No valid Adriatic files found (column detection failed).")
        return

    # ---------- PASS 1: compute GLOBAL DJ centroids across ALL files ----------
    dj_points = {k: [] for k in DJ_SET}

    for (path, time_col, lat_col, lon_col, station_col, basin_col) in file_specs:
        print("\n=== Scanning for DJ centroids:", path.name, "===")

        usecols = {time_col, lat_col, lon_col, station_col, basin_col}
        reader = pd.read_csv(
            path,
            chunksize=CHUNKSIZE,
            dtype=str,
            low_memory=False,
            usecols=lambda c: c in usecols
        )

        n_kept = 0
        for chunk in reader:
            basin_raw = chunk[basin_col].map(norm)
            basin_dj = basin_raw.map(remap_adr3_to_dj)

            m_dj = basin_dj.isin(DJ_SET)
            if not m_dj.any():
                continue

            dj_chunk = chunk.loc[m_dj].copy()
            # attach remapped dj labels
            dj_chunk["_dj"] = basin_dj.loc[m_dj].to_numpy()

            casts = compute_cast_positions(dj_chunk, station_col, time_col, lat_col, lon_col)

            dj_chunk["_cast"] = build_cast_id(dj_chunk, station_col, time_col)
            cast_dj = dj_chunk.groupby("_cast", as_index=False)["_dj"].first()
            casts = casts.merge(cast_dj, on="_cast", how="left")

            for dj in DJ_SET:
                sub = casts[casts["_dj"] == dj]
                if not sub.empty:
                    dj_points[dj].append(sub[["lat_med", "lon_med"]].to_numpy())
                    n_kept += len(sub)

        print(f"  DJ casts collected from this file: {n_kept}")

    centroids = {}
    for dj in sorted(DJ_SET):
        if len(dj_points[dj]) == 0:
            continue
        arr = np.vstack(dj_points[dj])
        centroids[dj] = (float(np.nanmedian(arr[:, 0])), float(np.nanmedian(arr[:, 1])))

    print("\n=== GLOBAL DJ centroids (lat, lon) ===")
    for dj in sorted(DJ_SET):
        if dj in centroids:
            clat, clon = centroids[dj]
            print(f"  {dj}: {clat:.4f}, {clon:.4f}")
        else:
            print(f"  {dj}: [MISSING]")

    if len(centroids) < 2:
        print("\n[ERROR] Not enough DJ centroids found. This suggests BASIN_3ADR is not nadr/madr/sadr in these files,")
        print("or the basin column is not what we think it is. Quick check: print unique BASIN_3ADR values from one file.")
        return

    # ---------- PASS 2: process each file, reassign OUT ----------
    for (path, time_col, lat_col, lon_col, station_col, basin_col) in file_specs:
        print("\n=== Processing:", path.name, "===")

        usecols = {time_col, lat_col, lon_col, station_col, basin_col}
        reader2 = pd.read_csv(
            path,
            chunksize=CHUNKSIZE,
            dtype=str,
            low_memory=False,
            usecols=lambda c: c in usecols
        )

        out_to_fixed = {}  # cast -> (DJ, dist_km)

        for chunk in reader2:
            basin_raw = chunk[basin_col].map(norm)
            m_out = basin_raw.str.lower().eq("out")
            if not m_out.any():
                continue

            out_chunk = chunk.loc[m_out].copy()
            casts = compute_cast_positions(out_chunk, station_col, time_col, lat_col, lon_col)

            lat = casts["lat_med"].to_numpy(dtype=float)
            lon = casts["lon_med"].to_numpy(dtype=float)

            dmat = []
            labels = []
            for dj in sorted(centroids.keys()):
                clat, clon = centroids[dj]
                dmat.append(haversine_km(lat, lon, clat, clon))
                labels.append(dj)

            dmat = np.vstack(dmat)
            imin = np.argmin(dmat, axis=0)
            dmin = dmat[imin, np.arange(dmat.shape[1])]
            fixed = np.array([labels[i] for i in imin], dtype=object)

            casts["_fixed"] = fixed
            casts["_dist_km"] = dmin

            out_to_fixed.update(dict(zip(casts["_cast"].to_numpy(),
             zip(casts["_fixed"].to_numpy(), casts["_dist_km"].astype(float).to_numpy()))))


        print(f"  OUT casts found: {len(out_to_fixed)}")
        if len(out_to_fixed) > 0:
            dists = np.array([v[1] for v in out_to_fixed.values()], dtype=float)
            print(f"  OUT->DJ dist_km: median={np.nanmedian(dists):.2f}  p90={np.nanpercentile(dists,90):.2f}  max={np.nanmax(dists):.2f}")

        # Write full file with BASIN_3ADR_FIXED appended
        out_path = path.with_name(path.stem + "_fixed.csv")
        print("  Writing:", out_path)

        full_reader = pd.read_csv(path, chunksize=CHUNKSIZE, low_memory=False)
        first = True

        for chunk in full_reader:
            if station_col not in chunk.columns or time_col not in chunk.columns or basin_col not in chunk.columns:
                chunk.to_csv(out_path, mode="w" if first else "a", index=False, header=first)
                first = False
                continue

            cast_id = chunk[station_col].astype(str) + "|" + chunk[time_col].astype(str)
            basin_raw = chunk[basin_col].map(norm)

            # Start with remapped DJ where possible (nadr->DJ1 etc), else keep original
            fixed = basin_raw.map(remap_adr3_to_dj)

            # Overwrite only OUT casts that have an assignment
            m_out = basin_raw.str.lower().eq("out")
            if m_out.any() and len(out_to_fixed) > 0:
                mapped = cast_id.map(lambda c: out_to_fixed.get(c, (None, None))[0])
                fixed.loc[m_out & mapped.notna()] = mapped.loc[m_out & mapped.notna()]

            chunk["BASIN_3ADR_FIXED"] = fixed

            chunk.to_csv(out_path, mode="w" if first else "a", index=False, header=first)
            first = False

        print("  Done:", out_path)

    print("\nAll done.")


if __name__ == "__main__":
    main()
