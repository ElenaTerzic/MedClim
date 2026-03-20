#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect SeaDataNet platform / instrument strings and (importantly) count at CAST level.

Why this exists:
- value_counts() on the instrument column counts ROWS (depth samples), not casts.
- For a fair comparison with your annual "casts per year" barplot, you need unique casts.

We define a cast id primarily by LOCAL_CDI_ID (preferred). If it's missing, we fall back to:
  Cruise | Station | YYYY-MM-DD

Outputs:
- prints detected column names
- prints cast counts by broad class (CTD / MBT-XBT / Bottle / TSG / Other)
- prints casts per year by class (head)
- prints the top raw strings in SDN_Other by unique-cast count

This is a streaming script (chunked) to handle very large SDN CSVs.
"""

from pathlib import Path
import pandas as pd
import numpy as np

SDN_CSV = Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins.csv")
CHUNKSIZE = 2_000_000  # tune for your RAM
MAX_CHUNKS = None      # set e.g. 10 for a quick test; None = full file

INSTR_COL = "Instrument / gear type:METAVAR:INDEXED_TEXT"
TIME_COL  = "yyyy-mm-ddThh:mm:ss.sss"
CRUISE_COL = "Cruise"
STATION_COL = "Station:METAVAR:INDEXED_TEXT"
CDI_COL = "LOCAL_CDI_ID"


def classify_sdn_platform(raw: str) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "SDN_Other"
    s = str(raw).strip().lower()

    if "thermosalinograph" in s:
        return "SDN_TSG"

    if ("water sampl" in s) or ("discrete water sampler" in s) or ("continuous water sampler" in s) or ("bottle" in s):
        return "SDN_Bottle"

    if ("bathythermograph" in s) or ("xbt" in s) or ("mbt" in s):
        return "SDN_MBTXBT"

    if "ctd" in s:
        return "SDN_CTD"

    has_temp = "water temperature sensor" in s
    has_sal  = "salinity sensor" in s or "salinometer" in s
    has_pres = ("water pressure sensor" in s) or ("water pressure sensors" in s) or ("pressure sensor" in s)

    if has_temp and has_sal and has_pres:
        return "SDN_CTD"

    if has_temp and has_sal and any(k in s for k in ["fluorometer", "transmissometer", "oxygen", "dissolved gas", "pH sensor", "titrator"]):
        return "SDN_CTD"

    return "SDN_Other"


def main():
    if not SDN_CSV.exists():
        raise FileNotFoundError(SDN_CSV)

    # Quick header scan
    head = pd.read_csv(SDN_CSV, nrows=5, dtype=str)
    print("\n=== SDN: columns in file ===")
    print(f"Total columns: {len(head.columns)}")
    # Show likely platform/instrument cols
    print("\n=== Candidate columns (matching keywords) ===")
    for c in head.columns:
        cl = c.lower()
        if any(k in cl for k in ["instrument", "gear", "platform", "device", "sensor"]):
            print(" -", c)

    need = [INSTR_COL, TIME_COL, CRUISE_COL, STATION_COL, CDI_COL]
    missing = [c for c in need if c not in head.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    usecols = need
    print("\nReading columns:", usecols)
    print("File:", SDN_CSV)

    # We'll accumulate unique (cast_id, year, class, raw) rows on disk to avoid huge RAM sets.
    tmp = SDN_CSV.parent / "tmp_sdn_cast_class_pairs.csv"
    if tmp.exists():
        tmp.unlink()

    n_chunks = 0
    for chunk in pd.read_csv(SDN_CSV, usecols=usecols, chunksize=CHUNKSIZE, dtype=str, low_memory=False):
        n_chunks += 1
        if (MAX_CHUNKS is not None) and (n_chunks > MAX_CHUNKS):
            break

        # parse year
        t = pd.to_datetime(chunk[TIME_COL], errors="coerce")
        chunk["_year"] = t.dt.year

        # cast id
        cdi = chunk[CDI_COL].astype(str).str.strip()
        cdi = cdi.where(~cdi.isin(["", "nan", "None"]), other=pd.NA)

        date = t.dt.date.astype(str)
        fallback = chunk[CRUISE_COL].astype(str) + "|" + chunk[STATION_COL].astype(str) + "|" + date
        chunk["_cast_id"] = cdi.fillna(fallback)

        # class
        chunk["_class"] = chunk[INSTR_COL].map(classify_sdn_platform)

        out = chunk[["_cast_id", "_year", "_class", INSTR_COL]].dropna(subset=["_cast_id", "_year"])
        out = out.drop_duplicates()

        out.to_csv(tmp, mode="a", index=False, header=(n_chunks == 1))

        if n_chunks % 5 == 0:
            print(f"  processed chunks: {n_chunks}")

    print("\nWrote unique cast-class pairs to:", tmp)

    df = pd.read_csv(tmp, dtype={"_year": "int64"})
    print("\n=== Unique casts by broad class ===")
    casts_by_class = df.groupby("_class")["_cast_id"].nunique().sort_values(ascending=False)
    print(casts_by_class.to_string())

    print("\n=== Casts per year by class (first 20 rows) ===")
    casts_year = (df.groupby(["_year", "_class"])["_cast_id"].nunique()
                    .reset_index(name="n_casts")
                    .sort_values(["_year", "_class"]))
    print(casts_year.head(20).to_string(index=False))

    print("\n=== Top SDN_Other raw strings by UNIQUE CAST count (top 30) ===")
    other = df[df["_class"] == "SDN_Other"]
    if len(other):
        top_other = other.groupby(INSTR_COL)["_cast_id"].nunique().sort_values(ascending=False).head(30)
        print(top_other.to_string())
    else:
        print("(No SDN_Other casts.)")

    print("\nDone.")

if __name__ == "__main__":
    main()
