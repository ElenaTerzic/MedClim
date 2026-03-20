#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_monthly_timeseries_composed_from_atomic.py

Build monthly time series for COMPOSED Mediterranean basins from the
atomic monthly time series file.

Input
-----
monthly_timeseries_all_bins.csv

Expected columns:
- basin
- z_m
- var
- time
- year
- month
- value
- n

Output
------
monthly_timeseries_all_bins_composed.csv

Aggregation logic
-----------------
For each composed basin × z_m × var × year × month:

    value_composed = sum(value_i * n_i) / sum(n_i)
    n_composed     = sum(n_i)

where i runs over the member atomic basins available for that month/bin/var.

Notes
-----
- This is the correct place to aggregate for composed-basin trends:
  aggregate monthly series first, then re-fit trends on the composed series.
- Do NOT average already-fitted trend coefficients or p-values afterwards.
- The script prefers explicit COMPOSED_BASINS below.
- If USE_BASINS_MANCA=True and Basins.manca is importable, it can also try
  to retrieve members from there.

Run
---
python build_monthly_timeseries_composed_from_atomic.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ==========================================================
# USER CONFIG
# ==========================================================

IN_CSV = "/Users/elena/Projects/IRB/MedClim/Results/Trends/monthly_timeseries_all_bins.csv"
OUT_CSV = "/Users/elena/Projects/IRB/MedClim/Results/Trends/monthly_timeseries_all_bins_composed.csv"

# Prefer this explicit dictionary first.
# Adjust if your exact composed naming differs.
COMPOSED_BASINS: Dict[str, List[str]] = {
    "ALB":  ["DS1"],
    "WEST": ["DS2", "DS3", "DS4", "DI1"],
    "NWM":  ["DF1", "DF2", "DF3", "DF4"],
    "TYR":  ["DT1", "DT2", "DT3"],
    "SIC":  ["DI3", "DJ4", "DJ5"],
    "ION":  ["DJ6", "DJ7", "DJ8", "DH1"],
    "LEV":  ["DL1", "DL2", "DL3", "DL4", "DH2", "DH3"],
    "NADR": ["DJ1"],
    "MADR": ["DJ2"],
    "SADR": ["DJ3"],
}

# Optional: try Basins.manca if available.
# If a composed basin name below is missing from the explicit dict,
# the script will try to obtain its members automatically.
USE_BASINS_MANCA = True

# If True, keep only atomic basin codes of pattern D?? (e.g. DF1, DJ3, DL2)
STRICT_ATOMIC_FILTER = True

# Sorting
SORT_OUTPUT = True


# ==========================================================
# HELPERS
# ==========================================================

def is_atomic_basin(code: str) -> bool:
    c = str(code).strip().upper()
    return len(c) == 3 and c.startswith("D") and c[1].isalpha() and c[2].isdigit()


def get_members_from_basins_manca(basin: str) -> List[str]:
    """
    Try to recover composed-basin members from Basins.manca.<name>,
    similar in spirit to your plot_manca_all_basins.py helper.
    """
    b = str(basin).strip().lower()

    try:
        import Basins  # type: ignore
    except Exception:
        return []

    try:
        if not hasattr(Basins, "manca") or not hasattr(Basins.manca, b):
            return []

        obj = getattr(Basins.manca, b)

        for attr in ("members", "basins", "children", "components"):
            if hasattr(obj, attr):
                mem = getattr(obj, attr)
                out = []
                for m in mem:
                    if hasattr(m, "abbrev"):
                        out.append(str(m.abbrev).upper())

                out = sorted(set(out))
                if STRICT_ATOMIC_FILTER:
                    out = [x for x in out if is_atomic_basin(x)]
                return out
    except Exception:
        return []

    return []


def resolve_composed_basins() -> Dict[str, List[str]]:
    """
    Start from explicit dictionary and optionally augment missing entries
    from Basins.manca.
    """
    out: Dict[str, List[str]] = {}

    # first: explicit
    for k, v in COMPOSED_BASINS.items():
        kk = str(k).upper().strip()
        vv = [str(x).upper().strip() for x in v]
        if STRICT_ATOMIC_FILTER:
            vv = [x for x in vv if is_atomic_basin(x)]
        out[kk] = sorted(set(vv))

    # optional: fill any empty/missing from Basins.manca
    if USE_BASINS_MANCA:
        keys = sorted(set(list(out.keys())))
        for k in keys:
            if len(out.get(k, [])) == 0:
                mem = get_members_from_basins_manca(k)
                if mem:
                    out[k] = mem

    return out


def weighted_mean(v: np.ndarray, w: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)

    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan

    return float(np.sum(v[m] * w[m]) / np.sum(w[m]))


def load_monthly(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])

    required = ["basin", "z_m", "var", "time", "year", "month", "value", "n"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["basin"] = df["basin"].astype(str).str.upper().str.strip()
    df["var"] = df["var"].astype(str).str.upper().str.strip()

    df["z_m"] = pd.to_numeric(df["z_m"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")

    df = df[
        df["basin"].notna() &
        df["var"].notna() &
        np.isfinite(df["z_m"]) &
        df["year"].notna() &
        df["month"].notna()
    ].copy()

    return df


def aggregate_one_composed(
    df_atomic: pd.DataFrame,
    composed_name: str,
    members: List[str],
) -> pd.DataFrame:
    """
    Aggregate atomic monthly time series into one composed basin.
    """
    members = [m.upper().strip() for m in members]
    d = df_atomic[df_atomic["basin"].isin(members)].copy()

    if d.empty:
        return pd.DataFrame(columns=["basin", "z_m", "var", "time", "year", "month", "value", "n"])

    gcols = ["z_m", "var", "time", "year", "month"]

    rows = []
    for key, g in d.groupby(gcols, dropna=False, sort=False):
        z_m, var, time, year, month = key

        v = g["value"].to_numpy(dtype=float)
        w = g["n"].to_numpy(dtype=float)

        val = weighted_mean(v, w)
        nsum = float(np.nansum(w[np.isfinite(w) & (w > 0)]))

        if not np.isfinite(val) or not np.isfinite(nsum) or nsum <= 0:
            continue

        rows.append({
            "basin": composed_name,
            "z_m": float(z_m),
            "var": str(var).upper(),
            "time": pd.Timestamp(time),
            "year": int(year),
            "month": int(month),
            "value": float(val),
            "n": float(nsum),
        })

    out = pd.DataFrame(rows)
    return out


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("[INFO] Reading:", IN_CSV)
    df = load_monthly(IN_CSV)

    composed = resolve_composed_basins()
    if not composed:
        raise RuntimeError("No composed basin definitions available.")

    atomic_present = sorted(df["basin"].dropna().unique())
    print(f"[INFO] Atomic basins present in input ({len(atomic_present)}): {atomic_present}")

    all_out = []

    for comp, members in composed.items():
        members_present = [m for m in members if m in set(atomic_present)]
        missing = [m for m in members if m not in set(atomic_present)]

        print(f"[INFO] {comp}:")
        print(f"       members declared = {members}")
        print(f"       members present  = {members_present}")
        if missing:
            print(f"       missing members  = {missing}")

        if len(members_present) == 0:
            print(f"[WARN] {comp}: no members found in input, skipping.")
            continue

        out_comp = aggregate_one_composed(df, comp, members_present)
        if out_comp.empty:
            print(f"[WARN] {comp}: aggregation produced no rows, skipping.")
            continue

        all_out.append(out_comp)
        print(f"       rows written     = {len(out_comp)}")

    if not all_out:
        raise RuntimeError("No composed monthly rows were produced.")

    df_out = pd.concat(all_out, ignore_index=True)

    if SORT_OUTPUT:
        df_out = df_out.sort_values(["var", "basin", "z_m", "time"]).reset_index(drop=True)

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    print("[OK] Wrote:", OUT_CSV)
    print(f"[OK] Rows={len(df_out)} | Basins={df_out['basin'].nunique()} | Bins={df_out['z_m'].nunique()}")

    print("[INFO] Output basins:")
    print(sorted(df_out["basin"].dropna().astype(str).str.upper().str.strip().unique()))


if __name__ == "__main__":
    main()