#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick SDN column / platform sanity checker.

Reads Seadatanet.profiles_only_with_basins.csv and prints:
- candidate columns related to instrument/gear/platform,
- top values + counts for the chosen platform column,
- how many casts would fall into the broad SDN classes used in the plots.

This is meant to help you understand why so much ends up as "SDN_Other".

Run:
  python inspect_sdn_columns_platforms.py

Edit SDN_PATH if needed.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

SDN_PATH = Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins.csv")

# Candidate keywords (case-insensitive)
KEYWORDS = [
    "instrument", "gear", "nstr", "platform", "device", "sensor",
    "probe", "ctd", "xbt", "mbt", "bottle", "rosette", "tsg",
]

def detect_platform_col(cols):
    preferred = [
        "Instrument / gear type:METAVAR:INDEXED_TEXT",
        "nstr -> Instrument / gear type:METAVAR:INDEXED_TEXT",
        "nstr",
    ]
    for p in preferred:
        if p in cols:
            return p
    low = [c.lower() for c in cols]
    for key in ["instrument / gear type", "gear type", "instrument", "nstr"]:
        for c, lc in zip(cols, low):
            if key in lc:
                return c
    return None

def classify_sdn_platform(val: str) -> str:
    s = str(val).strip().upper()
    if s in ("", "NAN", "NONE"):
        return "SDN_Other"
    if "CTD" in s:
        return "SDN_CTD"
    if ("XBT" in s) or ("MBT" in s):
        return "SDN_MBT/XBT"
    if ("BOTTLE" in s) or ("NISKIN" in s) or ("ROSETTE" in s):
        return "SDN_Bottle"
    if ("TSG" in s) or ("THERMOSALINOGRAPH" in s):
        return "SDN_TSG"
    return "SDN_Other"

def main():
    if not SDN_PATH.exists():
        raise FileNotFoundError(f"Missing: {SDN_PATH}")

    # Read only the header first
    cols = pd.read_csv(SDN_PATH, nrows=0).columns.tolist()
    print("\n=== SDN: columns in file ===")
    print(f"Total columns: {len(cols)}")

    print("\n=== Candidate columns (matching keywords) ===")
    candidates = []
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in KEYWORDS):
            candidates.append(c)
    for c in candidates:
        print(" -", c)

    plat_col = detect_platform_col(cols)
    print("\n=== Detected platform/instrument column ===")
    print("Selected:", plat_col)

    # Read only needed columns (limit load)
    usecols = [c for c in [plat_col, "Cruise", "Station:METAVAR:INDEXED_TEXT", "LOCAL_CDI_ID", "yyyy-mm-ddThh:mm:ss.sss"] if c and (c in cols)]
    if not usecols:
        # fallback: just read first 5 cols
        usecols = cols[:5]
    print("\nReading columns:", usecols)

    df = pd.read_csv(SDN_PATH, usecols=usecols, dtype=str, low_memory=False)
    print(f"Rows loaded: {len(df):,}")

    if plat_col and plat_col in df.columns:
        s = df[plat_col].astype(str).str.strip()
        s = s[~s.isna()]
        print("\n=== Top instrument/platform values (raw) ===")
        top = s.value_counts(dropna=False).head(30)
        for k, v in top.items():
            print(f"{v:10d}  {k}")

        print("\n=== Broad SDN class breakdown (heuristic) ===")
        cls = s.map(classify_sdn_platform)
        print(cls.value_counts().to_string())

        print("\nTip: If SDN_Other is huge, we can refine classify_sdn_platform() with your most common 'Other' strings.")
    else:
        print("\nWARNING: No platform/instrument column found/read. Nothing to summarize.")

    print("\nDone.\n")

if __name__ == "__main__":
    main()
