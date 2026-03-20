#!/usr/bin/env python3
"""
check_odv_time_format.py

Checks the completeness/format of the ODV-exported date/time column
(e.g. 'yyyy-mm-ddThh:mm:ss.sss') and reports how many records have
year-only vs year-month vs full dates/times.

Usage:
  python check_odv_time_format.py /path/to/your.csv
  python check_odv_time_format.py /path/to/your.csv --datecol "yyyy-mm-ddThh:mm:ss.sss"
"""

import argparse
import re
import sys
import pandas as pd


def guess_date_column(columns):
    # Common ODV/Medatlas-ish names and patterns
    preferred = [
        "yyyy-mm-ddThh:mm:ss.sss",
        "yyyy-mm-ddThh:mm:ss",
        "yyyy-mm-dd",
        "DATE",
        "Date",
        "TIME",
        "Time",
        "Datetime",
        "datetime",
    ]
    for c in preferred:
        if c in columns:
            return c

    # Pattern-based fallback (ODV sometimes embeds format in header)
    patt = re.compile(r"yyyy.*mm.*dd|Thh:mm:ss", re.IGNORECASE)
    matches = [c for c in columns if patt.search(c)]
    return matches[0] if matches else None


def categorize(s: str) -> str:
    if s is None:
        return "missing"
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return "missing"

    # Year only: 1991
    if re.fullmatch(r"\d{4}", s):
        return "year_only"

    # Year-month: 1991-10 or 1991/10
    if re.fullmatch(r"\d{4}[-/]\d{2}", s):
        return "year_month"

    # Date only: 1991-10-12 or 1991/10/12
    if re.fullmatch(r"\d{4}[-/]\d{2}[-/]\d{2}", s):
        return "date_only"

    # Compact date: 19911012
    if re.fullmatch(r"\d{8}", s):
        return "date_compact"

    # Datetime: 1991-10-12T03:04 or 1991-10-12T03:04:05(.sss)
    if re.fullmatch(
        r"\d{4}[-/]\d{2}[-/]\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?",
        s,
    ):
        return "datetime"

    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="ODV-exported CSV path")
    ap.add_argument("--datecol", default=None, help="Override date/time column name")
    ap.add_argument("--stationcol", default="Station", help="Station column name (default: Station)")
    ap.add_argument("--sample_only", action="store_true",
                    help="If set, only check rows that look like station metadata rows (Station not empty). "
                         "Recommended for ODV profile exports.")
    args = ap.parse_args()

    # Read as strings to avoid pandas coercing "1991" -> 1991-01-01 later
    try:
        df = pd.read_csv(args.csv, dtype=str)
    except Exception as e:
        print(f"ERROR reading CSV: {e}", file=sys.stderr)
        sys.exit(2)

    datecol = args.datecol or guess_date_column(df.columns)
    if not datecol or datecol not in df.columns:
        print("ERROR: Could not find a date/time column. Available columns are:", file=sys.stderr)
        for c in df.columns:
            print(f"  - {c}", file=sys.stderr)
        sys.exit(3)

    # For ODV CTD profiles, the date is typically only on the station metadata row,
    # with subsequent sample rows blank in meta columns.
    if args.sample_only:
        if args.stationcol not in df.columns:
            print(f"ERROR: stationcol '{args.stationcol}' not found. Use --stationcol.", file=sys.stderr)
            sys.exit(4)
        mask = df[args.stationcol].notna() & (df[args.stationcol].astype(str).str.strip() != "")
        sub = df.loc[mask, [datecol]].copy()
    else:
        sub = df[[datecol]].copy()

    values = sub[datecol].tolist()
    cats = [categorize(v) for v in values]

    # Summary
    total = len(cats)
    counts = pd.Series(cats).value_counts(dropna=False)
    print(f"\nDate/time column: {datecol}")
    print(f"Rows checked: {total:,}")
    print("\nCategory counts:")
    for k, v in counts.items():
        pct = (v / total * 100.0) if total else 0.0
        print(f"  {k:12s} {v:10,d}  ({pct:6.2f}%)")

    # Show a few examples for each non-empty category
    print("\nExamples (up to 5 each):")
    s = pd.Series(values)
    c = pd.Series(cats)
    for cat in ["year_only", "year_month", "date_only", "date_compact", "datetime", "other", "missing"]:
        idx = c[c == cat].index[:5]
        if len(idx) == 0:
            continue
        ex = [str(s[i]) for i in idx]
        print(f"  {cat:12s}: {ex}")

    # Detect “defaulted” dates like 01-01, which can hide missing precision
    # (only meaningful if some entries look like full dates)
    def looks_defaulted(x):
        x = str(x).strip()
        m = re.match(r"^(\d{4})[-/](\d{2})[-/](\d{2})", x)
        return bool(m and m.group(2) == "01" and m.group(3) == "01")

    defaulted = [v for v in values if v and str(v).strip() and looks_defaulted(v)]
    if defaulted:
        print(f"\nPotential defaulted dates (month=01 and day=01): {len(defaulted):,} "
              f"({len(defaulted)/total*100:.2f}% of checked rows)")
        print(f"  Example(s): {defaulted[:5]}")
    print("")


if __name__ == "__main__":
    main()
