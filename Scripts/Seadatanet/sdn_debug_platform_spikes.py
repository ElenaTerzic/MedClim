#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SDN debug: find suspicious casts that can create huge TSG/Sensors spikes.

Outputs:
- sdn_debug_spikes_cast_summary.csv (one row per cast)
- sdn_debug_spikes_cast_details_topN.csv (row-level excerpt for top offenders)
"""

from pathlib import Path
import numpy as np
import pandas as pd

SDN_PATH = Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins.csv")
CHUNKSIZE = 400_000

TIME_COL  = "yyyy-mm-ddThh:mm:ss.sss"
DEPTH_COL = "Depth [m]"
TEMP_COL  = "ITS-90 water temperature [degrees C]"
BASIN_COL = "basin"

CRUISE_COL  = "Cruise"
STATION_COL = "Station:METAVAR:INDEXED_TEXT"
LOCAL_COL   = "LOCAL_CDI_ID"

INSTR_COL = "Instrument / gear type:METAVAR:INDEXED_TEXT"
TQC_COL   = "QV:SEADATANET.1"
GOOD_QF   = {"1", "2"}

# threshold for "merged cast" suspicion
MIN_ROWS_SUSPECT = 5000
MIN_ZSPAN_SUSPECT = 800.0  # meters
MIN_UNIQUE_DEPTHS_SUSPECT = 300

def classify_sdn_row(instr: str) -> str:
    s = (instr or "").strip().lower()
    if "ctd" in s and "expendable" not in s:
        return "SDN_CTD"
    if "expendable ctd" in s:
        return "SDN_MBTXBT"
    if "bathythermograph" in s:
        return "SDN_MBTXBT"
    if "thermosalinograph" in s:
        return "SDN_TSG"
    if "discrete water sampler" in s or "continuous water sampler" in s:
        return "SDN_Bottle"
    if "water temperature sensor" in s or "salinity sensor" in s:
        return "SDN_Sensors"
    return "SDN_Other"

def resolve_cast_class(classes):
    s = set(classes)
    if "SDN_CTD" in s: return "SDN_CTD"
    if "SDN_MBTXBT" in s: return "SDN_MBTXBT"
    if "SDN_Bottle" in s: return "SDN_Bottle"
    if "SDN_TSG" in s: return "SDN_TSG"
    if "SDN_Sensors" in s: return "SDN_Sensors"
    return "SDN_Other"

def qc_mask(df):
    if TQC_COL not in df.columns:
        return np.ones(len(df), bool)
    q = df[TQC_COL].astype(str).str.strip().to_numpy()
    return np.isin(q, list(GOOD_QF))

def main():
    if not SDN_PATH.exists():
        raise FileNotFoundError(SDN_PATH)

    usecols = [
        TIME_COL, DEPTH_COL, TEMP_COL, BASIN_COL,
        CRUISE_COL, STATION_COL, LOCAL_COL,
        INSTR_COL, TQC_COL
    ]

    out_rows = []
    # to collect row-level excerpt for the worst offenders
    top_details = []

    reader = pd.read_csv(SDN_PATH, chunksize=CHUNKSIZE, dtype=str,
                         usecols=lambda c: c in set(usecols), low_memory=False)

    carry = None
    for chunk in reader:
        if carry is not None and len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)

        t = pd.to_datetime(chunk[TIME_COL], errors="coerce")
        chunk["_date"] = t.dt.date.astype(str)

        pid = chunk[CRUISE_COL].astype(str) + "|" + chunk[STATION_COL].astype(str) + "|" + chunk[LOCAL_COL].astype(str) + "|" + chunk["_date"].astype(str)
        chunk["_profile"] = pid

        last_prof = chunk["_profile"].iloc[-1]
        carry = chunk[chunk["_profile"] == last_prof].copy()
        chunk = chunk[chunk["_profile"] != last_prof]
        if len(chunk) == 0:
            continue

        for prof, g in chunk.groupby("_profile", sort=False):
            # QC + numeric parsing
            z = pd.to_numeric(g[DEPTH_COL], errors="coerce").to_numpy(float)
            T = pd.to_numeric(g[TEMP_COL], errors="coerce").to_numpy(float)
            m = np.isfinite(z) & np.isfinite(T) & qc_mask(g)

            if not np.any(m):
                continue

            z0 = z[m]
            zmin = float(np.nanmin(z0))
            zmax = float(np.nanmax(z0))
            zspan = zmax - zmin
            nrows = int(len(g))
            nvalid = int(np.sum(m))
            nuniqz = int(pd.Series(z0).round(3).nunique())

            instr_list = g[INSTR_COL].astype(str).tolist()
            row_classes = [classify_sdn_row(x) for x in instr_list]
            cast_class = resolve_cast_class(row_classes)

            # how mixed is this cast?
            class_counts = pd.Series(row_classes).value_counts()
            topclass = class_counts.index[0]
            frac_top = float(class_counts.iloc[0] / len(row_classes))

            # suspect criteria
            suspect = (
                (nrows >= MIN_ROWS_SUSPECT) or
                (zspan >= MIN_ZSPAN_SUSPECT) or
                (nuniqz >= MIN_UNIQUE_DEPTHS_SUSPECT)
            )

            if suspect or cast_class in ("SDN_TSG", "SDN_Sensors"):
                basin = str(g[BASIN_COL].iloc[0])
                # try to infer year quickly from date
                y = pd.to_datetime(g[TIME_COL].iloc[0], errors="coerce")
                year = int(y.year) if not pd.isna(y) else -1

                out_rows.append({
                    "profile": prof,
                    "basin": basin,
                    "year": year,
                    "cast_class_resolved": cast_class,
                    "top_rowclass": topclass,
                    "frac_top_rowclass": frac_top,
                    "nrows": nrows,
                    "nvalid_T": nvalid,
                    "zmin": zmin,
                    "zmax": zmax,
                    "zspan": zspan,
                    "n_unique_depths": nuniqz,
                    "rowclass_counts": class_counts.to_dict()
                })

    # finalize
    df = pd.DataFrame(out_rows)
    if df.empty:
        print("No suspicious casts found (unexpected).")
        return

    df = df.sort_values(["cast_class_resolved","nrows","zspan"], ascending=[True, False, False])
    out1 = Path("sdn_debug_spikes_cast_summary.csv")
    df.to_csv(out1, index=False)
    print("Wrote:", out1, "rows:", len(df))

    # Write row-level details for top offenders (by nrows) for deep debugging
    topN = 20
    worst = df.sort_values("nrows", ascending=False).head(topN)["profile"].tolist()
    print("\nTop offenders by nrows:")
    print(df.sort_values("nrows", ascending=False).head(10)[
        ["profile","basin","year","cast_class_resolved","nrows","zspan","n_unique_depths","frac_top_rowclass"]
    ])

    # Extract row excerpts for those topN profiles (small sample)
    # (Re-read file but only keep those profiles; still chunked)
    keep = set(worst)
    details = []
    reader = pd.read_csv(SDN_PATH, chunksize=CHUNKSIZE, dtype=str, low_memory=False)
    for chunk in reader:
        if TIME_COL not in chunk.columns:
            continue
        t = pd.to_datetime(chunk[TIME_COL], errors="coerce")
        chunk["_date"] = t.dt.date.astype(str)
        if all(c in chunk.columns for c in [CRUISE_COL, STATION_COL, LOCAL_COL]):
            chunk["_profile"] = chunk[CRUISE_COL].astype(str) + "|" + chunk[STATION_COL].astype(str) + "|" + chunk[LOCAL_COL].astype(str) + "|" + chunk["_date"].astype(str)
        else:
            continue
        sub = chunk[chunk["_profile"].isin(keep)]
        if len(sub):
            # keep only a few relevant cols if present
            cols = [c for c in [TIME_COL,BASIN_COL,DEPTH_COL,TEMP_COL,INSTR_COL,TQC_COL,CRUISE_COL,STATION_COL,LOCAL_COL,"_profile"] if c in sub.columns]
            details.append(sub[cols])

    if details:
        dd = pd.concat(details, ignore_index=True)
        out2 = Path("sdn_debug_spikes_cast_details_topN.csv")
        dd.to_csv(out2, index=False)
        print("Wrote:", out2, "rows:", len(dd))
    else:
        print("Could not extract details for top offenders (unexpected).")

if __name__ == "__main__":
    main()
