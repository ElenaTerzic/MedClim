#!/usr/bin/env python3
"""
SDN diagnostic table: station-level counts by instrument/gear type,
split by basin, period (20c/21c), and season.

This is designed to be comparable to SeaDataNet Table 3 (stations by instrument/gear type),
NOT raw-row counts.

Input: Seadatanet.profiles_only_with_basins.csv
Output: CSV summaries (and optionally prints top categories).
"""

import os
import re
import numpy as np
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
SDN_CSV = "/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins.csv"   # <-- adjust path
OUTDIR = "sdn_diagnostics"
os.makedirs(OUTDIR, exist_ok=True)

COL_TIME = "yyyy-mm-ddThh:mm:ss.sss"
COL_INST = "Instrument / gear type:METAVAR:INDEXED_TEXT"

# Choose ONE stable cast/station id strategy:
# 1) Prefer LOCAL_CDI_ID if present (often unique station id)
# 2) else fallback to Cruise+Station+time
CAND_ID_COLS = ["LOCAL_CDI_ID", "Cruise", "Station:METAVAR:INDEXED_TEXT", COL_TIME]

# Basin column name varies by your pipeline; try common ones:
BASIN_CANDIDATES = ["basin", "BASIN", "basin_abbrev", "BASIN_ABBREV", "subbasin", "SUBBASIN"]

# Define your periods (edit if you use different)
PERIODS = {
    "20c_1950_1999": (1950, 1999),
    "21c_2000_2019": (2000, 2019),
}

SEASONS = {
    "JFM": (1, 2, 3),
    "AMJ": (4, 5, 6),
    "JAS": (7, 8, 9),
    "OND": (10, 11, 12),
}

# ----------------------------
# Helpers
# ----------------------------

def find_first_present(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    return None

def parse_year_month(series):
    # robust: expects ISO-like; uses errors='coerce'
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.year.astype("Int64"), dt.dt.month.astype("Int64")

def classify_table3_like(raw: str) -> str:
    """
    Map SDN instrument/gear type strings to Table-3-like categories.
    Keep this intentionally transparent and easy to tweak.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "none info"
    s = str(raw).strip().lower()
    if s == "" or s in {"none", "nan"}:
        return "none info"

    # exact-ish keys first
    if "ctd" in s or "expendable ctd" in s:
        return "CTD"

    # bathythermographs (MBT / XBT family)
    if "bathythermograph" in s or "bathythermographs" in s or "xbt" in s or "mbt" in s:
        return "bathythermograph"

    # thermosalinograph / tsg
    if "thermosalinograph" in s or "thermosalinographs" in s or "tsg" in s:
        return "thermosalinograph"

    # bottle / discrete water sampler
    # (SDN often uses "discrete water samplers"; keep Table 3 wording)
    if "discrete water sampler" in s or "discrete water samplers" in s:
        return "discrete water sampler"
    if "continuous water sampler" in s or "continuous water samplers" in s:
        return "continuous water sampler"

    # thermistor chains (explicit Table 3 row)
    if "thermistor chain" in s or "thermistor chains" in s:
        return "thermistor chains"

    # explicit Table 3 row for sensor-pair
    # note: your data has EXACT "salinity sensor;water temperature sensor"
    if "salinity sensor" in s and "water temperature sensor" in s and ";" in s:
        # keep as its own label as per Table 3 screenshot
        return "salinity sensor; water temperature sensor"

    # salinometers (Table 3 row)
    if "salinometer" in s or "salinometers" in s:
        return "salinometers"

    # everything else
    return "Other"


def build_station_id(df: pd.DataFrame) -> pd.Series:
    if "LOCAL_CDI_ID" in df.columns:
        return df["LOCAL_CDI_ID"].astype(str)
    # fallback
    parts = []
    for c in ["Cruise", "Station:METAVAR:INDEXED_TEXT", COL_TIME]:
        if c in df.columns:
            parts.append(df[c].astype(str))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    return (parts[0] + "|" + parts[1] + "|" + parts[2]).astype(str)


# ----------------------------
# Main
# ----------------------------

def main():
    # Read header only first
    cols = pd.read_csv(SDN_CSV, nrows=0).columns.tolist()

    basin_col = find_first_present(cols, BASIN_CANDIDATES)
    if basin_col is None:
        raise SystemExit(
            f"Could not find a basin column. Tried: {BASIN_CANDIDATES}\n"
            f"Available columns: {cols}"
        )

    needed = [COL_INST, COL_TIME, basin_col]
    for c in ["LOCAL_CDI_ID", "Cruise", "Station:METAVAR:INDEXED_TEXT"]:
        if c in cols:
            needed.append(c)

    # Load only needed columns (much faster)
    df = pd.read_csv(SDN_CSV, usecols=needed, low_memory=False)

    # Parse time
    year, month = parse_year_month(df[COL_TIME])
    df["year"] = year
    df["month"] = month

    # Drop rows without valid time
    df = df[df["year"].notna() & df["month"].notna()].copy()

    # Classify instrument types (Table-3-like)
    df["inst_table3"] = df[COL_INST].map(classify_table3_like)

    # Station id for station-level counting
    df["station_id"] = build_station_id(df)

    # Assign season
    def month_to_season(m):
        for sn, mm in SEASONS.items():
            if int(m) in mm:
                return sn
        return None

    df["season"] = df["month"].map(month_to_season)
    df = df[df["season"].notna()].copy()

    # Assign period label
    def year_to_period(y):
        y = int(y)
        for pname, (y0, y1) in PERIODS.items():
            if y0 <= y <= y1:
                return pname
        return None

    df["period"] = df["year"].map(year_to_period)
    df = df[df["period"].notna()].copy()

    # --------
    # IMPORTANT: station-level counts
    # --------
    # Deduplicate to unique station_id within each grouping
    group_cols = [basin_col, "period", "season", "inst_table3"]
    station_counts = (
        df.drop_duplicates(subset=[basin_col, "period", "season", "inst_table3", "station_id"])
          .groupby(group_cols)["station_id"]
          .size()
          .reset_index(name="n_stations")
    )

    # Also compute raw-row counts (so you can see how far station vs row differ)
    row_counts = (
        df.groupby(group_cols)
          .size()
          .reset_index(name="n_rows")
    )

    out1 = os.path.join(OUTDIR, "sdn_table3_like_station_counts.csv")
    out2 = os.path.join(OUTDIR, "sdn_table3_like_row_counts.csv")
    station_counts.to_csv(out1, index=False)
    row_counts.to_csv(out2, index=False)

    # Convenience: pivot summary (station-level) for quick inspection
    pivot = station_counts.pivot_table(
        index="inst_table3",
        columns=["period"],
        values="n_stations",
        aggfunc="sum",
        fill_value=0
    ).sort_values(by=list(PERIODS.keys()), ascending=False)

    print("\n=== Basin column detected ===")
    print(basin_col)

    print("\n=== Station-level totals by instrument (all basins+seasons summed) ===")
    print(pivot)

    # Where does Other happen?
    other_break = (
        station_counts[station_counts["inst_table3"] == "Other"]
        .groupby(["period"])["n_stations"]
        .sum()
    )
    print("\n=== Total 'Other' stations by period (summed all basins+seasons) ===")
    print(other_break)

    print(f"\nWrote:\n - {out1}\n - {out2}\n")


if __name__ == "__main__":
    main()
