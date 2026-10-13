#!/usr/bin/env python3
# Reads : Data/Seadatanet/Seadatanet.profiles_only.csv
# Writes: Data/Seadatanet/Seadatanet.profiles_only_with_basins.csv
import os
import numpy as np
import pandas as pd
import Basins

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Scripts.paths import DATA

LON_COL   = "Longitude [degrees_east]"
LAT_COL   = "Latitude [degrees_north]"
BASIN_COL = "basin"

BASIN_LIST = [
    Basins.manca.df2, Basins.manca.df3, Basins.manca.df4, Basins.manca.ds2, Basins.manca.df1,
    Basins.manca.ds1, Basins.manca.ds3, Basins.manca.ds4, Basins.manca.dt1, Basins.manca.dt2,
    Basins.manca.dt3, Basins.manca.di1, Basins.manca.di3, Basins.manca.dj1, Basins.manca.dj2,
    Basins.manca.dj3, Basins.manca.dj6, Basins.manca.dj4, Basins.manca.dj7, Basins.manca.dj8,
    Basins.manca.dj5, Basins.manca.dh1, Basins.manca.dh2, Basins.manca.dh3, Basins.manca.dl1,
    Basins.manca.dl2, Basins.manca.dl3, Basins.manca.dl4
]

def add_basins_chunked(input_csv: str, out_csv: str, chunksize: int = 300_000) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    wrote_header = False
    total_in = total_geo = total_out = 0

    reader = pd.read_csv(
        input_csv,
        chunksize=chunksize,
        dtype=str,
        low_memory=False,
    )

    for i, chunk in enumerate(reader, start=1):
        total_in += len(chunk)

        if LON_COL not in chunk.columns or LAT_COL not in chunk.columns:
            raise ValueError(f"Missing lon/lat columns: {LON_COL}, {LAT_COL}")

        lon = pd.to_numeric(chunk[LON_COL], errors="coerce")
        lat = pd.to_numeric(chunk[LAT_COL], errors="coerce")
        good = lon.notna() & lat.notna()
        chunk = chunk.loc[good].copy()
        lon = lon.loc[good].astype(float).to_numpy()
        lat = lat.loc[good].astype(float).to_numpy()

        totalSZ = len(chunk)
        total_geo += RZ if (RZ := len(chunk)) else 0
        if RZ == 0:
            continue

        xyz = np.zeros((RZ, 3), dtype=float)
        xyz[:, 0] = lon
        xyz[:, 1] = lat

        basin_arr = np.array(["none"] * RZ, dtype=object)
        for basin in BASIN_LIST:
            mask = basin.areinside(xyz)
            if mask is not None:
                basin_arr[mask] = basin.abbrev

        chunk[BASIN_COL] = basin_arr
        chunk = chunk[chunk[BASIN_COL] != "none"]
        if len(chunk) == 0:
            continue

        chunk.to_csv(out_csv, mode="a", index=False, header=(not wrote_header))
        wrote_header = True
        total_out += len(chunk)

        if i % 10 == 0:
            print(f"[chunk {i}] read={total_in:,} geo_ok={total_geo:,} in_basins={total_out:,}")

    print("\nDone.")
    print(f"Input rows read:         {total_in:,}")
    print(f"Valid lon/lat rows:      {total_geo:,}")
    print(f"Rows assigned to basins: {total_out:,}")
    print(f"Wrote: {out_csv}")

if __name__ == "__main__":
    INFILE = str(DATA / "Seadatanet" / "Seadatanet.profiles_only.csv")
    OUTFILE = str(DATA / "Seadatanet" / "Seadatanet.profiles_only_with_basins.csv")
    add_basins_chunked(INFILE, OUTFILE, chunksize=300_000)
