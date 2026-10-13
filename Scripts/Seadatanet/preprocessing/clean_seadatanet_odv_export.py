#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reads : Data/Seadatanet/Seadatanet.csv (ODV export)
# Writes: Data/Seadatanet/Seadatanet.clean.csv, Seadatanet.profiles_only.csv, Seadatanet.clean_issues.txt

import csv
from pathlib import Path
from collections import defaultdict

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Scripts.paths import DATA

# --------------------
# EDIT PATHS
# --------------------
INFILE = DATA / "Seadatanet" / "Seadatanet.csv"

OUT_CLEAN  = INFILE.with_suffix(".clean.csv")
OUT_MINI   = INFILE.with_suffix(".profiles_only.csv")
OUT_ISSUES = INFILE.with_suffix(".clean_issues.txt")

# --------------------
# CONFIG
# --------------------
SKIP_PREFIXES = ("//",)

DATE_COL    = "yyyy-mm-ddThh:mm:ss.sss"
LON_COL     = "Longitude [degrees_east]"
LAT_COL     = "Latitude [degrees_north]"
CAST_COL    = "LOCAL_CDI_ID"
DEPTH_COL   = "Depth [m]"
TEMP_COL    = "ITS-90 water temperature [degrees C]"
SAL_COL     = "Water body salinity [per mille]"
INST_COL    = "Instrument / gear type:METAVAR:INDEXED_TEXT"
CRUISE_COL  = "Cruise"
STATION_COL = "Station:METAVAR:INDEXED_TEXT"

# Desired QC names (we will CREATE .1/.2 if the raw header has duplicates)
DEPTH_QC_COL  = "QV:SEADATANET"
TEMP_QC_COL   = "QV:SEADATANET.1"
SAL_QC_COL    = "QV:SEADATANET.2"
SAMPLE_QC_COL = "QV:SEADATANET:SAMPLE"  # optional

def norm(s: str) -> str:
    """Normalize header strings to avoid hidden whitespace/BOM/NBSP mismatches."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\ufeff", "")     # BOM
    s = s.replace("\xa0", " ")      # NBSP
    return s.strip()

def make_unique_headers(cols):
    """
    Mimic pandas behavior for duplicate column names:
      X, X, X  ->  X, X.1, X.2
    """
    counts = defaultdict(int)
    out = []
    for c in cols:
        if counts[c] == 0:
            out.append(c)
        else:
            out.append(f"{c}.{counts[c]}")
        counts[c] += 1
    return out

BASE_COLS = [
    CRUISE_COL, STATION_COL, DATE_COL, LON_COL, LAT_COL,
    CAST_COL, DEPTH_COL, TEMP_COL, SAL_COL, INST_COL
]

# Require temp/sal QC columns (after we disambiguate duplicates)
QC_REQUIRED = [TEMP_QC_COL, SAL_QC_COL]
QC_OPTIONAL = [DEPTH_QC_COL, SAMPLE_QC_COL]

def main():
    print(f"Input:   {INFILE}")
    print(f"Clean:   {OUT_CLEAN}")
    print(f"Mini:    {OUT_MINI}")
    print(f"Issues:  {OUT_ISSUES}")

    n_lines = 0
    n_skipped_history = 0
    n_bad_fieldcount = 0
    n_written = 0

    with INFILE.open("r", encoding="utf-8", errors="replace", newline="") as fin, \
         OUT_CLEAN.open("w", encoding="utf-8", newline="") as fout_clean, \
         OUT_MINI.open("w", encoding="utf-8", newline="") as fout_mini, \
         OUT_ISSUES.open("w", encoding="utf-8") as fissues:

        reader = csv.reader(fin, delimiter=",", quotechar='"')
        w_clean = csv.writer(fout_clean, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w_mini  = csv.writer(fout_mini,  delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # ---- read header (first non-history line) ----
        header_raw = None
        while True:
            row = next(reader, None)
            if row is None:
                raise RuntimeError("File ended before header was found.")
            n_lines += 1
            if row and str(row[0]).startswith(SKIP_PREFIXES):
                n_skipped_history += 1
                continue
            header_raw = row
            break

        # normalize header
        header_norm = [norm(c) for c in header_raw]

        # IMPORTANT: disambiguate duplicates (pandas-style)
        header = make_unique_headers(header_norm)

        ncols = len(header)
        col2i = {c: i for i, c in enumerate(header)}

        print(f"Detected ncols = {ncols}")

        # Debug: show QV columns after disambiguation
        qv_cols = [c for c in header if c.startswith("QV:")]
        print("QV columns found in header (after making unique):")
        for c in qv_cols:
            print("  ", repr(c))

        # Normalize expected names too
        base_cols    = [norm(c) for c in BASE_COLS]
        qc_required  = [norm(c) for c in QC_REQUIRED]
        qc_optional  = [norm(c) for c in QC_OPTIONAL]

        missing_base = [c for c in base_cols if c not in col2i]
        if missing_base:
            raise KeyError(f"Missing required BASE columns in header: {missing_base}")

        missing_req_qc = [c for c in qc_required if c not in col2i]
        if missing_req_qc:
            raise KeyError(
                "Missing required QC columns in header: "
                f"{missing_req_qc}\n"
                "Tip: the file likely does not include distinct temp/sal QC columns."
            )

        # Build mini columns
        mini_cols = base_cols.copy()

        # Insert depth QC right after depth if present
        if norm(DEPTH_QC_COL) in col2i:
            i = mini_cols.index(norm(DEPTH_COL)) + 1
            mini_cols.insert(i, norm(DEPTH_QC_COL))

        # Insert temp QC right after TEMP
        i = mini_cols.index(norm(TEMP_COL)) + 1
        mini_cols.insert(i, norm(TEMP_QC_COL))

        # Insert sal QC right after SAL
        i = mini_cols.index(norm(SAL_COL)) + 1
        mini_cols.insert(i, norm(SAL_QC_COL))

        # Append sample QC if present
        if norm(SAMPLE_QC_COL) in col2i:
            mini_cols.append(norm(SAMPLE_QC_COL))

        mini_idx = [col2i[c] for c in mini_cols]

        qc_kept = [c for c in mini_cols if c.startswith("QV:")]
        print(f"Keeping {len(mini_cols)} mini columns.")
        print(f"QC kept: {qc_kept}")

        # ---- define which columns must NOT be forward-filled ----
        meas_idx = set()
        for i, name in enumerate(header):
            if name in (norm(DEPTH_COL), norm(TEMP_COL), norm(SAL_COL)) or name.startswith("QV:"):
                meas_idx.add(i)

        ffill_idx = [i for i in range(ncols) if i not in meas_idx]
        last = [""] * ncols

        # Write headers (UNIQUE header to clean + mini so downstream scripts see .1/.2)
        w_clean.writerow(header)
        w_mini.writerow(mini_cols)

        # ---- process rows ----
        for row in reader:
            n_lines += 1
            if not row:
                continue

            if str(row[0]).startswith(SKIP_PREFIXES):
                n_skipped_history += 1
                continue

            if len(row) != ncols:
                n_bad_fieldcount += 1
                if n_bad_fieldcount <= 2000:
                    fissues.write(f"[FIELD_COUNT_MISMATCH] line {n_lines}: got {len(row)} expected {ncols}\n")
                    fissues.write(f"  row_head = {row[:8]}\n")
                continue

            # Forward fill metadata columns only
            for i in ffill_idx:
                if row[i] == "":
                    row[i] = last[i]
                else:
                    last[i] = row[i]

            # Keep only rows with a depth value
            if row[col2i[norm(DEPTH_COL)]] == "":
                continue

            w_clean.writerow(row)
            w_mini.writerow([row[i] for i in mini_idx])
            n_written += 1

            if n_written % 2_000_000 == 0:
                print(f"  wrote {n_written:,} profile-rows... (lines read {n_lines:,})")

    print("\nDone.")
    print(f"Lines read:            {n_lines:,}")
    print(f"Skipped history lines: {n_skipped_history:,}")
    print(f"Bad fieldcount rows:   {n_bad_fieldcount:,} (logged first 2000)")
    print(f"Rows written:          {n_written:,}")
    print("\nNext step: use *.profiles_only.csv for stats/climatology (fast).")
    print("IMPORTANT: define profiles by LOCAL_CDI_ID in downstream scripts.")

if __name__ == "__main__":
    main()
