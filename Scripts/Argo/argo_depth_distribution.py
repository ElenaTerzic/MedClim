#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# =========================
# USER CONFIG (ARGO ONLY)
# =========================
ARGO_CSV = Path("/Users/elena/Projects/IRB/MedClim/Data/Argo/CSV/Argo_with_basins.csv")

TIME_COL     = "TIME"
PLATFORM_COL = "PLATFORM_NUMBER"
CYCLE_COL    = "CYCLE_NUMBER"
DEPTH_COL    = "PRES"   # dbar ~ meters for these purposes

CHUNKSIZE = 400_000

DEPTH_THRESHOLDS = [200, 500, 1000, 1500, 2000, 3000, 4000, 5000]
DEPTH_BANDS = [(0, 200), (200, 1000), (1000, 2000), (2000, 4000), (4000, 6000)]
ROUND_LEVELS = [1.0, 5.0, 10.0]

# cap dz storage (sampling) to keep memory bounded
DZ_SAMPLE_MAX = 2_000_000
RNG_SEED = 0

# outlier diagnostics
TOPK_OUTLIERS = 30
NLEV_OUTLIER_THRESHOLDS = [500, 1000, 2000, 5000]


# =========================
# PROFILE STREAMER
# =========================
def _profile_id_from_chunk(chunk: pd.DataFrame) -> pd.Series:
    t = pd.to_datetime(chunk[TIME_COL], errors="coerce")
    day = t.dt.date.astype(str)
    return (
        chunk[PLATFORM_COL].astype(str) + "|" +
        chunk[CYCLE_COL].astype(str) + "|" +
        day.astype(str)
    )

def iter_profiles_argo(csv_path: Path, chunksize: int = CHUNKSIZE):
    """
    Yields (pid, group_df) where pid = PLATFORM|CYCLE|date.
    Assumes rows for a given pid are contiguous in file (typical for exports).
    """
    usecols = [TIME_COL, PLATFORM_COL, CYCLE_COL, DEPTH_COL]
    usecols_set = set(usecols)

    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        dtype=str,
        low_memory=False,
        usecols=lambda c: c in usecols_set
    )

    carry = None
    for chunk in reader:
        if carry is not None and len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)

        for c in usecols:
            if c not in chunk.columns:
                raise ValueError(f"Missing required column: {c}")

        chunk["_pid"] = _profile_id_from_chunk(chunk)

        last_pid = chunk["_pid"].iloc[-1]
        carry = chunk[chunk["_pid"] == last_pid].copy()
        chunk = chunk[chunk["_pid"] != last_pid]
        if len(chunk) == 0:
            continue

        for pid, g in chunk.groupby("_pid", sort=False):
            yield pid, g

    if carry is not None and len(carry):
        carry["_pid"] = _profile_id_from_chunk(carry)
        for pid, g in carry.groupby("_pid", sort=False):
            yield pid, g


# =========================
# HELPERS
# =========================
def pct(a: np.ndarray, p: float) -> float:
    return float(np.nanpercentile(a, p)) if a.size else np.nan

def _append_dz_sample(store: list, dz: np.ndarray, rng: np.random.Generator):
    """Bounded random replacement (good enough for diagnostics)."""
    if dz.size == 0:
        return
    room = DZ_SAMPLE_MAX - len(store)
    if room > 0:
        take = dz[:room]
        store.extend(take.astype(float).tolist())
        dz = dz[room:]
    if dz.size == 0:
        return
    n = len(store)
    if n == 0:
        return
    for val in dz:
        j = int(rng.integers(0, n))
        store[j] = float(val)

def _top_k_dict(d: dict, k: int = 25):
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]


# =========================
# ANALYSIS
# =========================
def analyze_argo_native_depth_grid(csv_path: Path):
    print("\n" + "=" * 92)
    print("ARGO NATIVE DEPTH GRID FORENSICS (NO STD DEPTH ARRAY)")
    print(f"File: {csv_path}")
    print("=" * 92)

    n_profiles = 0
    n_valid_profiles = 0
    n_with_2lev = 0
    n_surface_only = 0

    zmax_sample = []
    nlev_sample = []

    cov_counts = {thr: 0 for thr in DEPTH_THRESHOLDS}

    # common depth levels (rounded, per-profile unique)
    common_counts = {r: {} for r in ROUND_LEVELS}

    # dz overall + by band (sampled)
    rng = np.random.default_rng(RNG_SEED)
    dz_all = []
    dz_by_band = {band: [] for band in DEPTH_BANDS}

    # outlier tracking (top K by nlev)
    outliers = []  # list of tuples: (nlev, zmax, pid)
    outlier_counts = Counter()

    # quick grouping sanity: count missing cycle occurrences
    pid_cycle_missing = 0
    pid_total = 0

    for pid, g in iter_profiles_argo(csv_path):
        n_profiles += 1

        pid_total += 1
        # heuristic: missing cycle often appears as 'nan' or empty in pid
        parts = str(pid).split("|")
        if len(parts) >= 2:
            cyc = parts[1].strip().lower()
            if cyc in ("nan", "none", ""):
                pid_cycle_missing += 1

        z = pd.to_numeric(g[DEPTH_COL], errors="coerce").to_numpy(dtype=float)
        z = z[np.isfinite(z)]
        if z.size == 0:
            continue

        # sort & unique
        z = np.unique(np.sort(z))
        nlev = int(z.size)
        zmax = float(z[-1])

        n_valid_profiles += 1
        nlev_sample.append(nlev)
        zmax_sample.append(zmax)

        # after you compute z = np.unique(np.sort(z)) and nlev, zmax
        if nlev >= 2000:  # or maybe >= 1000
            z01 = np.unique(np.round(z / 0.1) * 0.1).size
            z05 = np.unique(np.round(z / 0.5) * 0.5).size
            z10 = np.unique(np.round(z / 1.0) * 1.0).size
            print(f"OUTLIER_ROUNDING pid={pid} nlev_exact={nlev} nlev@0.1={z01} nlev@0.5={z05} nlev@1.0={z10} zmax={zmax:.1f}")


        if nlev >= 2:
            n_with_2lev += 1
        else:
            n_surface_only += 1

        for thr in DEPTH_THRESHOLDS:
            if zmax >= thr:
                cov_counts[thr] += 1

        # count nlev outliers
        for thr in NLEV_OUTLIER_THRESHOLDS:
            if nlev >= thr:
                outlier_counts[thr] += 1

        # maintain top-K outliers
        outliers.append((nlev, zmax, pid))
        if len(outliers) > (TOPK_OUTLIERS * 5):
            outliers.sort(key=lambda t: t[0], reverse=True)
            outliers = outliers[:TOPK_OUTLIERS * 2]

        # most common depths (rounded) — count each rounded depth once per profile
        for r in ROUND_LEVELS:
            zr = np.round(z / r) * r
            zr = np.unique(zr)
            dct = common_counts[r]
            for val in zr:
                dct[val] = dct.get(val, 0) + 1

        # dz overall + by band
        if nlev >= 2:
            dz = np.diff(z)
            dz = dz[np.isfinite(dz) & (dz > 0)]
            _append_dz_sample(dz_all, dz, rng)

            mids = 0.5 * (z[:-1] + z[1:])
            for (a, b) in DEPTH_BANDS:
                m = (mids >= a) & (mids < b)
                if np.any(m):
                    _append_dz_sample(dz_by_band[(a, b)], dz[m], rng)

    # -------------------------
    # PRINT REPORT
    # -------------------------
    print(f"\nProfiles streamed (raw groups): {n_profiles:,}")
    print(f"Profiles with at least 1 valid depth: {n_valid_profiles:,}")
    if n_valid_profiles == 0:
        print("No valid profiles found. Check file/path and required columns.")
        return

    print(f"Profiles with >=2 depth levels: {n_with_2lev:,} ({100*n_with_2lev/n_valid_profiles:.1f}%)")
    print(f"Surface-only (1 level): {n_surface_only:,} ({100*n_surface_only/n_valid_profiles:.1f}%)")

    if pid_total:
        print(f"\nProfile-ID sanity: CYCLE missing in pid: {pid_cycle_missing:,}/{pid_total:,} ({100*pid_cycle_missing/pid_total:.2f}%)")

    zmax_arr = np.asarray(zmax_sample, float)
    nlev_arr = np.asarray(nlev_sample, float)

    print("\nDepth reach (zmax) summary [m~dbar]:")
    print(f"  median={pct(zmax_arr,50):.1f}  p10={pct(zmax_arr,10):.1f}  p90={pct(zmax_arr,90):.1f}  max={np.nanmax(zmax_arr):.1f}")

    print("\nNumber of levels per profile:")
    print(f"  median={pct(nlev_arr,50):.0f}  p10={pct(nlev_arr,10):.0f}  p90={pct(nlev_arr,90):.0f}  max={np.nanmax(nlev_arr):.0f}")

    if outlier_counts:
        print("\nHigh-nlevels counts:")
        for thr in NLEV_OUTLIER_THRESHOLDS:
            print(f"  nlevels >= {thr:5d}: {outlier_counts[thr]:,} profiles ({100*outlier_counts[thr]/n_valid_profiles:.3f}%)")

    print("\nDepth coverage (fraction reaching at least ...):")
    for thr in DEPTH_THRESHOLDS:
        print(f"  >= {thr:4d} m : {cov_counts[thr]:,} ({100*cov_counts[thr]/n_valid_profiles:.1f}%)")

    def print_dz_stats(name: str, arr_list: list):
        if not arr_list:
            print(f"\nΔz stats ({name}): no samples")
            return
        a = np.asarray(arr_list, float)
        print(f"\nΔz (vertical step) stats — {name} [m]:")
        print(f"  median={pct(a,50):.3f}  p10={pct(a,10):.3f}  p90={pct(a,90):.3f}  max={np.nanmax(a):.3f}")
        a2 = np.round(a, 2)
        vals, cnts = np.unique(a2, return_counts=True)
        top = np.argsort(cnts)[::-1][:15]
        print("  Most common Δz (rounded to 0.01 m):")
        for i in top:
            print(f"    {vals[i]:7.2f} m : {cnts[i]:,}")

    print_dz_stats("ALL DEPTHS", dz_all)
    for (a, b) in DEPTH_BANDS:
        print_dz_stats(f"{a}-{b} m", dz_by_band[(a, b)])

    print("\nMost common depth levels (counted as 'profiles containing that level'), top 25:")
    for r in ROUND_LEVELS:
        d = common_counts[r]
        items = _top_k_dict(d, k=25)
        print(f"\n  Rounded to {r:g} m:")
        for val, cnt in items:
            print(f"    {val:7.1f} m : {cnt:,} profiles ({100*cnt/n_valid_profiles:.1f}%)")

    # Top outliers
    outliers.sort(key=lambda t: t[0], reverse=True)
    outliers = outliers[:TOPK_OUTLIERS]
    print(f"\nTOP {TOPK_OUTLIERS} PROFILES BY #LEVELS (after unique-depth compression):")
    for nlev, zmax, pid in outliers:
        print(f"  nlev={nlev:6d}  zmax={zmax:7.1f}  pid={pid}")


def main():
    if not ARGO_CSV.exists():
        raise FileNotFoundError(f"Missing file: {ARGO_CSV}")
    analyze_argo_native_depth_grid(ARGO_CSV)

if __name__ == "__main__":
    main()
