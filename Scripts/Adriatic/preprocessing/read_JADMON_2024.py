#!/usr/bin/env python3
"""
Read the JADMON 2024 Adriatic CTD delivery into the standard long format.

Companion to ``read_CTD_19_23.py``.  JADMON 2024 continues the same national
monitoring programme as CTD_HR_2019_2023_MFSD: 50 of its 70 stations match
CTD_HR station coordinates exactly, and the two do not overlap in time
(CTD_HR ends 2023, this delivery covers 01.01.2024-31.12.2024), so there is no
duplication between them.  It is kept as a separate SOURCE so that its effect
on the climatologies can be tested, and so the dataset list required by
Reviewer 1 stays honest about provenance.

Input format
------------
Two tab-separated files, one per variable, each with:
  lines 0-3   metadata header (parameter, project, period)
  lines 5-75  station table: POSTAJA, OPIS, LAT, LNG
  line  78    data header:   postaja, vrijeme, dubina, vrijednost, VF, KAT
  lines 79+   data, with postaja and vrijeme given only on the first row of
              each cast and blank thereafter

Temperature and salinity are reported at identical station/time/depth keys, so
every level carries both - unlike much of the historical record, where many
casts are temperature-only.

Output
------
Data/Adriatic/Processed/JADMON_2024_long.csv, schema identical to
CTD_HR_2019_2023_MFSD_long.csv:
    TIME, LONGITUDE, LATITUDE, STATION, BASIN_3ADR, DEPTH_M, TEMP_C, SAL_PSU, SOURCE

The downstream mainland/coastal filtering and the BASIN_3ADR_FIXED assignment
are left to the existing pipeline (remove_coastline_data.py and the outlier
fixing step), exactly as for the other Adriatic sources.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

import Basins

# --------------------------------------------------
# INPUT
# --------------------------------------------------
INDIR = Path("/Users/elena/Work/IRB/Podaci/Jadran_2024")
TEMP_FILE = INDIR / "temperatura_2024.csv"
SAL_FILE = INDIR / "salinitet_2024.csv"

OUTDIR = Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed")
OUT_CSV = OUTDIR / "JADMON_2024_long.csv"

DIAGDIR = Path("/Users/elena/Projects/IRB/MedClim/Diagnostics/JADMON_2024")

STATION_HEADER_ROW = 5
STATION_LAST_ROW = 76
DATA_HEADER_ROW = 78

SOURCE_NAME = "HR_JADMON_CSV"
STATION_PREFIX = "JADMON"

# Only keep values the provider flagged as good.  The 2024 delivery is
# entirely VF=1, but future deliveries may not be.
GOOD_FLAGS = [1]


# --------------------------------------------------
# READING
# --------------------------------------------------

def read_one(path: Path, value_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (station table, long data frame) for one variable file."""
    raw = path.read_text(encoding="utf-8", errors="replace").split("\n")

    stations = pd.read_csv(
        io.StringIO("\n".join(raw[STATION_HEADER_ROW:STATION_LAST_ROW])), sep="\t")
    stations = stations.dropna(subset=["POSTAJA", "LAT", "LNG"])

    data = pd.read_csv(io.StringIO("\n".join(raw[DATA_HEADER_ROW:])), sep="\t")
    # station and time appear once per cast and are blank on following rows
    data["postaja"] = data.postaja.ffill()
    data["vrijeme"] = data.vrijeme.ffill()
    data = data.rename(columns={"vrijednost": value_name})

    data = data[data.VF.isin(GOOD_FLAGS)]
    data = data.dropna(subset=["postaja", "vrijeme", "dubina", value_name])

    return stations, data[["postaja", "vrijeme", "dubina", value_name]]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    DIAGDIR.mkdir(parents=True, exist_ok=True)

    stations, temp = read_one(TEMP_FILE, "TEMP_C")
    stations_s, sal = read_one(SAL_FILE, "SAL_PSU")

    if not stations.equals(stations_s):
        print("WARNING: station tables differ between the two files")

    print("--- INPUT ---")
    print(f"stations listed : {len(stations)}")
    print(f"temperature rows: {len(temp):,}")
    print(f"salinity rows   : {len(sal):,}")

    # Repeated depths within a cast: the delivery gives date only, with no time
    # of day, so repeat occupations of a station on one day cannot be told
    # apart and are averaged.
    for name, d in (("temperature", temp), ("salinity", sal)):
        dup = d.duplicated(["postaja", "vrijeme", "dubina"]).sum()
        print(f"{name}: {dup:,} repeated station/date/depth rows, averaged")
    temp = temp.groupby(["postaja", "vrijeme", "dubina"], as_index=False).TEMP_C.mean()
    sal = sal.groupby(["postaja", "vrijeme", "dubina"], as_index=False).SAL_PSU.mean()

    df = temp.merge(sal, on=["postaja", "vrijeme", "dubina"], how="outer")
    df["TIME"] = pd.to_datetime(df.vrijeme, format="%d.%m.%Y", errors="coerce")
    df = df.merge(stations, left_on="postaja", right_on="POSTAJA", how="left")

    missing = df.LAT.isna().sum()
    if missing:
        print(f"WARNING: {missing:,} rows without station coordinates, dropped")
    df = df.dropna(subset=["TIME", "LAT", "LNG", "dubina"])
    df = df.dropna(subset=["TEMP_C", "SAL_PSU"], how="all")

    # --------------------------------------------------
    # BASIN ASSIGNMENT  (same polygons and ordering as read_CTD_19_23.py)
    # --------------------------------------------------
    basin_list = [Basins.adriatic.north, Basins.adriatic.mid, Basins.adriatic.south]

    def basins_abbrev_for_lonlat(lon, lat, basin_list, default="out"):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        out = np.full(lon.shape, default, dtype=object)
        valid = np.isfinite(lon) & np.isfinite(lat)
        if not np.any(valid):
            return out
        xyz = np.zeros((valid.sum(), 3), dtype=float)
        xyz[:, 0] = lon[valid]
        xyz[:, 1] = lat[valid]
        idx_valid = np.where(valid)[0]
        for basin in basin_list:
            mask = basin.areinside(xyz)
            out[idx_valid[mask]] = basin.abbrev
        return out

    df["BASIN_3ADR"] = basins_abbrev_for_lonlat(df.LNG.values, df.LAT.values, basin_list)

    df["STATION"] = (STATION_PREFIX + "_"
                     + df.LNG.round(5).astype(str) + "_"
                     + df.LAT.round(5).astype(str))

    out = pd.DataFrame({
        "TIME": df.TIME,
        "LONGITUDE": df.LNG,
        "LATITUDE": df.LAT,
        "STATION": df.STATION,
        "BASIN_3ADR": df.BASIN_3ADR,
        "DEPTH_M": df.dubina,
        "TEMP_C": df.TEMP_C,
        "SAL_PSU": df.SAL_PSU,
        "SOURCE": SOURCE_NAME,
    }).sort_values(["TIME", "STATION", "DEPTH_M"])

    out.to_csv(OUT_CSV, index=False)

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------
    casts = out.groupby(["STATION", out.TIME.dt.date]).agg(
        n=("DEPTH_M", "size"), zmax=("DEPTH_M", "max"))

    print("\n--- OUTPUT ---")
    print(f"rows written    : {len(out):,}")
    print(f"unique stations : {out.STATION.nunique()}")
    print(f"casts           : {len(casts)}")
    print(f"time range      : {out.TIME.min():%Y-%m-%d} to {out.TIME.max():%Y-%m-%d}")
    print(f"depth range     : {out.DEPTH_M.min():.1f} to {out.DEPTH_M.max():.1f} m")
    print(f"levels with both T and S: "
          f"{(out.TEMP_C.notna() & out.SAL_PSU.notna()).mean() * 100:.1f}%")
    print("\nbasin assignment (rows):")
    print(out.BASIN_3ADR.value_counts().to_string())
    print("\ncasts by basin and depth reached:")
    bas = out.groupby(["STATION", out.TIME.dt.date]).BASIN_3ADR.first()
    summary = pd.DataFrame({"basin": bas, "zmax": casts.zmax})
    print(summary.groupby("basin").zmax.describe()[["count", "min", "50%", "max"]]
          .round(1).to_string())
    for z in (200, 600, 1000):
        deep = summary[summary.zmax >= z]
        print(f"  casts reaching >={z:5d} m: {len(deep):3d}  "
              f"({deep.basin.value_counts().to_dict()})")

    print(f"\nmonthly cast counts:")
    print(casts.reset_index().groupby(
        pd.to_datetime(casts.reset_index()['TIME']).dt.month).size().to_string())

    summary.to_csv(DIAGDIR / "JADMON_2024_cast_summary.csv")
    print(f"\nSaved: {OUT_CSV}")
    print(f"Saved: {DIAGDIR / 'JADMON_2024_cast_summary.csv'}")


if __name__ == "__main__":
    main()
