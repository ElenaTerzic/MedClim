#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SeaDataNet: detect "standard depth levels" per platform and compare 20c vs 21c

Outputs:
  - cast_summaries_by_period_platform.csv
  - depth_occurrence_by_period_platform.csv
  - standard_depths_by_period_platform.csv   (depths appearing in >= THRESH_FRAC of casts)
  - depth_occurrence_curves_<period>_0-2000m.png
  - dz_hist_<period>.png                     (hist of per-cast dz_median)

Key definitions
---------------
- cast id: LOCAL_CDI_ID (preferred), else Cruise|Station|date fallback
- depth occurrence: counts casts having that depth (rounded to 1 m), using UNIQUE depths per cast
- "standard depths": depths whose occurrence fraction >= THRESH_FRAC within a group
- dz per cast: median of successive depth differences (sorted unique rounded depths)

Notes
-----
- If your file is not roughly grouped by cast id, streaming-by-cast will be wrong.
  (Your cleaned ODV export is usually grouped, so this should be fine.)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# USER CONFIG
# ---------------------------
INFILE = Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only.csv")
OUTDIR = Path("/Users/elena/Projects/IRB/MedClim/Plots/Seadatanet") / "DepthGrid"
OUTDIR.mkdir(parents=True, exist_ok=True)

# If you have basins file, it's OK; basins are not required here.
# INFILE = Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins.csv")

DATE_COL   = "yyyy-mm-ddThh:mm:ss.sss"
DEPTH_COL  = "Depth [m]"
CAST_COL   = "LOCAL_CDI_ID"
CRUISE_COL = "Cruise"
STATION_COL= "Station:METAVAR:INDEXED_TEXT"
INST_COL   = "Instrument / gear type:METAVAR:INDEXED_TEXT"

CHUNKSIZE = 500_000

# Periods to compare
PERIODS = {
    "20c_1950_1999": (1950, 1999),
    "21c_2000_2019": (2000, 2019),
}

# Round depths for "standard levels"
DEPTH_ROUND_M = 1.0

# Consider depths only within this range for the "standard depths" table/plots
PLOT_ZMAX = 2000.0

# Define "standard depth levels" as those appearing in at least this fraction of casts
THRESH_FRAC = 0.05  # 5% of casts within (period, platform)

# For very large datasets, keep only a sample of per-cast dz/zmax/nlevels to compute quantiles
SAMPLE_MAX = 300_000

# Hist range for dz plots (meters)
DZ_HIST_MAX = 250.0
DZ_HIST_BINS = 80


# ---------------------------
# Platform classification
# ---------------------------
def classify_platform(instr: str) -> str:
    """
    Heuristic mapping from instrument string to platform class.
    Adjust keywords as needed for your SDN export.
    """
    s = (instr or "").strip().lower()

    if s == "" or s in ("nan", "none"):
        return "Other"

    # TSG / thermo-salinograph
    if "tsg" in s or "thermosalinograph" in s or "thermo-salinograph" in s:
        return "TSG"

    # Bottle / rosette / niskin
    if "bottle" in s or "niskin" in s or "rosette" in s or "sampler" in s:
        return "Bottle"

    # XBT / MBT
    if "xbt" in s or "mbt" in s or "bathythermograph" in s:
        return "MBT/XBT"

    # CTD (include common strings)
    if "ctd" in s or "seabird" in s or "sea-bird" in s or "sbe" in s:
        return "CTD"

    return "Other"


# ---------------------------
# Cast-id fallback (if LOCAL_CDI_ID missing)
# ---------------------------
def build_cast_id(df: pd.DataFrame) -> pd.Series:
    cid = df.get(CAST_COL)
    if cid is None:
        cid = pd.Series([""] * len(df), index=df.index)

    cid = cid.astype(str).str.strip()
    has = cid.ne("") & cid.ne("nan") & cid.ne("None")

    # fallback cast key: Cruise|Station|YYYY-MM-DD
    # (fast: ISO format year-month-day is first 10 chars)
    d = df[DATE_COL].astype(str).str.slice(0, 10).fillna("")
    fb = (
        df.get(CRUISE_COL, "").astype(str).fillna("") + "|" +
        df.get(STATION_COL, "").astype(str).fillna("") + "|" +
        d
    )
    return cid.where(has, fb)


def parse_year(date_str: str) -> Optional[int]:
    """
    Fast year parsing for ISO-like strings 'YYYY-...'
    """
    if not isinstance(date_str, str):
        return None
    if len(date_str) < 4:
        return None
    y = date_str[:4]
    if not y.isdigit():
        return None
    return int(y)


def which_period(year: int) -> Optional[str]:
    for label, (ymin, ymax) in PERIODS.items():
        if ymin <= year <= ymax:
            return label
    return None


# ---------------------------
# Streaming cast iterator
# ---------------------------
def iter_casts(csv_path: Path, chunksize: int) -> Iterator[Tuple[str, pd.DataFrame]]:
    """
    Stream CSV in chunks; keep the last cast in carry so cast groups
    do not split across chunk boundary.

    Assumes rows are roughly grouped by cast id.
    """
    usecols = [DATE_COL, DEPTH_COL, CAST_COL, CRUISE_COL, STATION_COL, INST_COL]

    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        dtype=str,
        low_memory=False,
        usecols=lambda c: c in set(usecols),
    )

    carry = None

    for chunk in reader:
        if carry is not None and len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)

        # ensure required
        for col in (DATE_COL, DEPTH_COL):
            if col not in chunk.columns:
                raise ValueError(f"Missing required column: {col}")

        # build cast
        if CAST_COL not in chunk.columns:
            chunk["_cast"] = build_cast_id(chunk)
        else:
            chunk["_cast"] = build_cast_id(chunk)

        last_cast = chunk["_cast"].iloc[-1]
        carry = chunk[chunk["_cast"] == last_cast].copy()
        chunk = chunk[chunk["_cast"] != last_cast]
        if len(chunk) == 0:
            continue

        for cast_id, g in chunk.groupby("_cast", sort=False):
            yield cast_id, g

    if carry is not None and len(carry):
        for cast_id, g in carry.groupby("_cast", sort=False):
            yield cast_id, g


# ---------------------------
# Reservoir sampler for per-cast metrics
# ---------------------------
class Reservoir:
    def __init__(self, k: int, seed: int = 0):
        self.k = int(k)
        self.rng = np.random.default_rng(seed)
        self.data: List[float] = []
        self.n_seen = 0

    def add(self, x: float):
        self.n_seen += 1
        if not np.isfinite(x):
            return
        if len(self.data) < self.k:
            self.data.append(float(x))
        else:
            j = self.rng.integers(0, self.n_seen)
            if j < self.k:
                self.data[j] = float(x)

    def arr(self) -> np.ndarray:
        return np.array(self.data, dtype=float)


@dataclass
class GroupAcc:
    n_casts: int = 0
    depth_counts: Counter = None
    dz_med: Reservoir = None
    zmax: Reservoir = None
    nlevels: Reservoir = None
    n_surface_only: int = 0

    def __post_init__(self):
        if self.depth_counts is None:
            self.depth_counts = Counter()
        if self.dz_med is None:
            self.dz_med = Reservoir(SAMPLE_MAX, seed=1)
        if self.zmax is None:
            self.zmax = Reservoir(SAMPLE_MAX, seed=2)
        if self.nlevels is None:
            self.nlevels = Reservoir(SAMPLE_MAX, seed=3)


def cast_depth_metrics(depths: np.ndarray) -> Tuple[float, float, int]:
    """
    depths: numeric depth array for this cast (meters)
    returns: (zmax, dz_median, nlevels)
    """
    z = depths[np.isfinite(depths)]
    if z.size == 0:
        return (np.nan, np.nan, 0)

    # round and unique
    z = np.round(z / DEPTH_ROUND_M) * DEPTH_ROUND_M
    z = np.unique(z)
    z = z[(z >= 0.0) & (z <= 6000.0)]
    if z.size == 0:
        return (np.nan, np.nan, 0)

    z.sort()
    zmax = float(z[-1])
    nlev = int(z.size)

    if z.size < 2:
        dzm = np.nan
    else:
        dz = np.diff(z)
        dz = dz[np.isfinite(dz) & (dz > 0)]
        dzm = float(np.median(dz)) if dz.size else np.nan

    return (zmax, dzm, nlev)


def quantiles(x: np.ndarray, qs=(0.25, 0.5, 0.75)) -> Tuple[float, ...]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return tuple(np.nan for _ in qs)
    return tuple(np.quantile(x, qs))


# ---------------------------
# Main processing
# ---------------------------
def main():
    print("Input:", INFILE)
    print("Outdir:", OUTDIR)
    print("Periods:", PERIODS)
    print("Depth round:", DEPTH_ROUND_M, "m")
    print("Standard depth threshold:", THRESH_FRAC * 100, "% of casts in group")

    groups: Dict[Tuple[str, str], GroupAcc] = defaultdict(GroupAcc)

    n_rows = 0
    n_casts = 0

    for k, (cast_id, g) in enumerate(iter_casts(INFILE, CHUNKSIZE)):
        n_casts += 1
        n_rows += len(g)

        # year/period from first row
        y = parse_year(str(g[DATE_COL].iloc[0]))
        if y is None:
            continue
        period = which_period(y)
        if period is None:
            continue

        instr = str(g[INST_COL].iloc[0]) if INST_COL in g.columns else ""
        platform = classify_platform(instr)

        # depths
        z = pd.to_numeric(g[DEPTH_COL], errors="coerce").to_numpy()
        zmax, dzm, nlev = cast_depth_metrics(z)
        if nlev == 0:
            continue

        acc = groups[(period, platform)]
        acc.n_casts += 1
        acc.zmax.add(zmax)
        acc.nlevels.add(float(nlev))
        if np.isfinite(dzm):
            acc.dz_med.add(dzm)

        if nlev <= 1:
            acc.n_surface_only += 1

        # update depth occurrence (unique rounded depths per cast)
        zz = np.round(z / DEPTH_ROUND_M) * DEPTH_ROUND_M
        zz = zz[np.isfinite(zz)]
        zz = zz[(zz >= 0.0) & (zz <= PLOT_ZMAX)]
        if zz.size:
            for d in np.unique(zz):
                acc.depth_counts[float(d)] += 1

        if (k % 2000) == 0 and k > 0:
            print(f"casts={n_casts:,} rows={n_rows:,} groups={len(groups):,}")

    # ---------------------------
    # Write summaries
    # ---------------------------
    # 1) per-group summary stats
    summary_rows = []
    for (period, platform), acc in sorted(groups.items()):
        z_arr = acc.zmax.arr()
        dz_arr = acc.dz_med.arr()
        nl_arr = acc.nlevels.arr()

        zq25, zq50, zq75 = quantiles(z_arr)
        dzq25, dzq50, dzq75 = quantiles(dz_arr)
        nq25, nq50, nq75 = quantiles(nl_arr)

        summary_rows.append({
            "period": period,
            "platform_class": platform,
            "n_casts": acc.n_casts,
            "zmax_q25_m": zq25, "zmax_median_m": zq50, "zmax_q75_m": zq75,
            "dz_q25_m": dzq25, "dz_median_m": dzq50, "dz_q75_m": dzq75,
            "nlevels_q25": nq25, "nlevels_median": nq50, "nlevels_q75": nq75,
            "frac_surface_only": (acc.n_surface_only / acc.n_casts) if acc.n_casts else np.nan,
            "dz_samples_seen": int(acc.dz_med.n_seen),
            "dz_samples_kept": int(len(acc.dz_med.data)),
        })

    df_sum = pd.DataFrame(summary_rows)
    out_sum = OUTDIR / "cast_summaries_by_period_platform.csv"
    df_sum.to_csv(out_sum, index=False)
    print("Wrote:", out_sum)

    # 2) depth occurrence table (casts having depth / n_casts in group)
    occ_rows = []
    for (period, platform), acc in sorted(groups.items()):
        if acc.n_casts == 0:
            continue
        for d, c in acc.depth_counts.items():
            occ_rows.append({
                "period": period,
                "platform_class": platform,
                "depth_rounded_m": d,
                "n_casts_with_depth": int(c),
                "frac_casts_with_depth": float(c) / float(acc.n_casts),
            })

    df_occ = pd.DataFrame(occ_rows)
    out_occ = OUTDIR / "depth_occurrence_by_period_platform.csv"
    df_occ.to_csv(out_occ, index=False)
    print("Wrote:", out_occ)

    # 3) "standard depths": depths with frac >= THRESH_FRAC
    df_std = df_occ[df_occ["frac_casts_with_depth"] >= THRESH_FRAC].copy()
    df_std = df_std.sort_values(["period", "platform_class", "depth_rounded_m"])
    out_std = OUTDIR / "standard_depths_by_period_platform.csv"
    df_std.to_csv(out_std, index=False)
    print("Wrote:", out_std)

    # ---------------------------
    # Plots
    # ---------------------------
    # A) depth occurrence curves (fraction vs depth) per period
    for period, (ymin, ymax) in PERIODS.items():
        plt.figure(figsize=(10, 8))

        sub = df_occ[(df_occ["period"] == period) & (df_occ["depth_rounded_m"] <= PLOT_ZMAX)]
        if sub.empty:
            plt.title(f"{period}: no data")
            plt.savefig(OUTDIR / f"depth_occurrence_curves_{period}_0-2000m.png", dpi=200)
            plt.close()
            continue

        for platform in ["CTD", "MBT/XBT", "Bottle", "TSG", "Other"]:
            s = sub[sub["platform_class"] == platform].sort_values("depth_rounded_m")
            if s.empty:
                continue
            plt.plot(s["frac_casts_with_depth"].to_numpy(), s["depth_rounded_m"].to_numpy(), label=platform)

        plt.gca().invert_yaxis()
        plt.xlabel("Fraction of casts having this depth (rounded to 1 m)")
        plt.ylabel("Depth (m)")
        plt.title(f"SeaDataNet: depth-level occurrence by platform ({period})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_png = OUTDIR / f"depth_occurrence_curves_{period}_0-2000m.png"
        plt.savefig(out_png, dpi=200)
        plt.close()
        print("Wrote:", out_png)

    # B) dz median per cast histograms per period (one panel per platform)
    for period in PERIODS.keys():
        platforms = ["CTD", "MBT/XBT", "Bottle", "TSG", "Other"]
        fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=False)
        axs = axs.ravel()

        for i, platform in enumerate(platforms):
            ax = axs[i]
            acc = groups.get((period, platform))
            if acc is None:
                ax.set_title(platform)
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                continue

            dz = acc.dz_med.arr()
            dz = dz[np.isfinite(dz) & (dz >= 0)]
            if dz.size == 0:
                ax.set_title(platform)
                ax.text(0.5, 0.5, "no dz samples", ha="center", va="center")
                continue

            # per-cast dz median: this avoids CTD dominating by number of points
            ax.hist(dz[dz <= DZ_HIST_MAX], bins=DZ_HIST_BINS)
            overflow = int(np.sum(dz > DZ_HIST_MAX))
            ax.set_title(platform)
            if overflow > 0:
                ax.text(0.98, 0.92, f"overflow (dz>{DZ_HIST_MAX:g}m): {overflow}",
                        transform=ax.transAxes, ha="right", va="top")

            ax.set_xlabel("Δz median per cast (m)")
            ax.set_ylabel("Cast count")

        # last subplot empty
        if len(axs) > len(platforms):
            axs[-1].axis("off")

        fig.suptitle(f"SeaDataNet: within-cast vertical spacing Δz (median per cast) — {period}")
        fig.tight_layout()
        out_png = OUTDIR / f"dz_hist_{period}.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print("Wrote:", out_png)

    print("\nDone.")
    print("Key files:")
    print(" -", out_sum)
    print(" -", out_occ)
    print(" -", out_std)


if __name__ == "__main__":
    main()
