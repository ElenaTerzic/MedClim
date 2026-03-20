#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot cast counts by depth bin as "vertical histograms" per basin.

- Rows: seasons (JFM/AMJ/JAS/OND)
- Cols: periods (21c_2000_2019, 20c_1950_1999)
- y-axis: depth (increasing downwards)
- x-axis: number of casts that contribute >=1 valid T sample in bin
- SDN: stacked bars (CTD, MBT/XBT, Bottle)   [TSG/Sensors/Other EXCLUDED]
- ARGO + ADRIATIC: markers (not stacked)

Key fix: SDN platform class is resolved PER CAST using all rows, with priority:
CTD > MBT/XBT > Bottle
Everything else is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =============================================================================
# CONFIG
# =============================================================================

PERIODS = {
    "21c_2000_2025": (2000, 2025),
    "20c_1950_1999": (1950, 1999),
}

SEASONS = {
    "JFM": {1, 2, 3},
    "AMJ": {4, 5, 6},
    "JAS": {7, 8, 9},
    "OND": {10, 11, 12},
}

STD_CENTERS = np.array([
    5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200,
    250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,
    1200, 1300, 1400, 1500, 1750, 2000, 2500, 3000,
    3500, 4000, 4500, 5000
], dtype=float)

PLOT_ZMAX = 2000.0

OUTDIR = Path("/Users/elena/Projects/IRB/MedClim/Plots/MultiPlatform/CastCounts")
OUTDIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 400_000

GOOD_QF_SDN = {"1", "2"}
GOOD_QF_ARGO = {"1", "2"}

ADR3_TO_MANCA_DJ = {
    "nadr": "DJ1",
    "madr": "DJ2",
    "sadr": "DJ3",
    "north": "DJ1",
    "mid": "DJ2",
    "middle": "DJ2",
    "central": "DJ2",
    "south": "DJ3",
}

SDN_INSTR_COL = "Instrument / gear type:METAVAR:INDEXED_TEXT"

# KEEP ONLY THESE THREE:
SDN_CLASSES = ["SDN_CTD", "SDN_MBTXBT", "SDN_Bottle"]

SDN_COLORS = {
    "SDN_CTD":    "#d62728",  # red
    "SDN_MBTXBT": "#ff7f0e",  # orange
    "SDN_Bottle": "#2ca02c",  # green
}

ARGO_COLOR = "black"
ADRI_COLOR = "#e377c2"  # pink-magenta


# =============================================================================
# BIN EDGES (centered)
# =============================================================================

def make_centered_bin_edges(centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = np.asarray(centers, float).copy()
    c.sort()
    mid = 0.5 * (c[:-1] + c[1:])
    first = 0.0
    last = c[-1] + 0.5 * (c[-1] - c[-2])
    edges = np.concatenate([[first], mid, [last]])
    return c, edges

BIN_CENTERS, BIN_EDGES = make_centered_bin_edges(STD_CENTERS)
NBINS = len(BIN_CENTERS)


# =============================================================================
# HELPERS
# =============================================================================

def month_to_season(m: int) -> Optional[str]:
    for s, months in SEASONS.items():
        if m in months:
            return s
    return None

def normalize_basin_value(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "nan"):
        return ""
    return s

def apply_basin_remap(basin: str, remap: Optional[Dict[str, str]]) -> str:
    if remap is None:
        return basin
    key = basin.strip().lower()
    return remap.get(key, basin)

def _qc_mask(df: pd.DataFrame, qc_col: Optional[str], good_qf: Optional[set]) -> np.ndarray:
    if qc_col is None or good_qf is None or qc_col not in df.columns:
        return np.ones(len(df), dtype=bool)
    q = df[qc_col].astype(str).str.strip().to_numpy()
    return np.isin(q, list(good_qf))

def classify_sdn_row(instr: str) -> str:
    """
    Row-level SDN instrument classification.
    KEEP ONLY: CTD / MBT-XBT / Bottle.
    Everything else -> SDN_IGNORE.
    """
    s = (instr or "").strip().lower()

    # CTD
    if "ctd" in s and "expendable" not in s:
        return "SDN_CTD"

    # MBT / XBT (incl expendable CTD)
    if "expendable ctd" in s:
        return "SDN_MBTXBT"
    if "bathythermograph" in s or "bathythermographs" in s:
        return "SDN_MBTXBT"
    if "expendable bathythermograph" in s or "expendable bathythermographs" in s:
        return "SDN_MBTXBT"
    if "mechanical bathythermograph" in s:
        return "SDN_MBTXBT"

    # Bottle / discrete sampling
    if "discrete water sampler" in s or "discrete water samplers" in s:
        return "SDN_Bottle"
    if "continuous water sampler" in s or "continuous water samplers" in s:
        return "SDN_Bottle"

    return "SDN_IGNORE"

def resolve_sdn_cast_class(row_classes: Iterable[str]) -> Optional[str]:
    """
    Cast-level resolution with priority.
    Returns None for casts we ignore (TSG / sensors / other).
    """
    s = set(row_classes)
    if "SDN_CTD" in s:
        return "SDN_CTD"
    if "SDN_MBTXBT" in s:
        return "SDN_MBTXBT"
    if "SDN_Bottle" in s:
        return "SDN_Bottle"
    return None

def bin_indices(z: np.ndarray) -> np.ndarray:
    bi = np.searchsorted(BIN_EDGES, z, side="right") - 1
    return bi


# =============================================================================
# SOURCE SPECS
# =============================================================================

@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path

    time_col: str
    basin_col: str
    depth_col: str
    temp_col: str

    profile_cols: List[str]

    temp_qc_col: Optional[str] = None
    depth_qc_col: Optional[str] = None
    good_qf: Optional[set] = None

    basin_remap: Optional[Dict[str, str]] = None
    depth_kind: str = "depth_m"  # "depth_m" or "pres_dbar"

    sdn_instr_col: Optional[str] = None


SOURCES: List[SourceSpec] = [
    SourceSpec(
        name="SDN",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins_mainland_special_excluded.csv"),
        time_col="yyyy-mm-ddThh:mm:ss.sss",
        basin_col="basin",
        depth_col="Depth [m]",
        temp_col="ITS-90 water temperature [degrees C]",
        profile_cols=["Cruise", "Station:METAVAR:INDEXED_TEXT", "LOCAL_CDI_ID"],
        temp_qc_col="QV:SEADATANET.1",
        good_qf=GOOD_QF_SDN,
        depth_kind="depth_m",
        sdn_instr_col=SDN_INSTR_COL,
    ),
    SourceSpec(
        name="ARGO",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Argo/CSV/Argo_with_basins_mainland_special_excluded.csv"),
        time_col="TIME",
        basin_col="basin",
        depth_col="PRES",
        temp_col="TEMP",
        profile_cols=["PLATFORM_NUMBER", "CYCLE_NUMBER"],
        temp_qc_col="TEMP_QC",
        depth_qc_col="PRES_QC",
        good_qf=GOOD_QF_ARGO,
        depth_kind="pres_dbar",
    ),

    # Adriatic local datasets (pooled into "ADRIATIC" at counting time)
    SourceSpec(
        name="CTD_HR_2019_2023_MFSD",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/CTD_HR_2019_2023_MFSD_long_fixed_mainland_special_excluded.csv"),
        time_col="TIME",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        profile_cols=["STATION"],
        basin_remap=ADR3_TO_MANCA_DJ,
        depth_kind="depth_m",
    ),
    SourceSpec(
        name="RB_Northern_Adriatic_Transect_1979_2018",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/RB_Northern_Adriatic_Transect_1979_2018_long_fixed_mainland_special_excluded.csv"),
        time_col="TIME",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        profile_cols=["STATION"],
        basin_remap=ADR3_TO_MANCA_DJ,
        depth_kind="depth_m",
    ),
    SourceSpec(
        name="Palagruza_Transect_1952_2018",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/Palagruza_Transect_1952_2018_long_fixed_mainland_special_excluded.csv"),
        time_col="TIME",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        profile_cols=["STATION"],
        basin_remap=ADR3_TO_MANCA_DJ,
        depth_kind="depth_m",
    ),
    SourceSpec(
        name="IOR_Adriatic_database_1987_2019",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/IOR_Adriatic_database_1987_2019_long_fixed_mainland_special_excluded.csv"),
        time_col="TIME",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        profile_cols=["STATION", "CAST_IDX"],
        basin_remap=ADR3_TO_MANCA_DJ,
        depth_kind="depth_m",
    ),
]


# =============================================================================
# CORE COUNTING
# =============================================================================

def iter_profiles_csv(spec: SourceSpec, chunksize: int = CHUNKSIZE) -> Iterable[pd.DataFrame]:
    if not spec.path.exists():
        raise FileNotFoundError(f"Missing input: {spec.path}")

    usecols = list(set(spec.profile_cols + [spec.time_col, spec.basin_col, spec.depth_col, spec.temp_col]))
    if spec.temp_qc_col:
        usecols.append(spec.temp_qc_col)
    if spec.depth_qc_col:
        usecols.append(spec.depth_qc_col)
    if spec.sdn_instr_col:
        usecols.append(spec.sdn_instr_col)

    reader = pd.read_csv(
        spec.path,
        chunksize=chunksize,
        dtype=str,
        usecols=lambda c: c in set(usecols),
        low_memory=False
    )

    carry = None
    for chunk in reader:
        if carry is not None and len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)

        t = pd.to_datetime(chunk[spec.time_col], errors="coerce")
        chunk["_date"] = t.dt.date.astype(str)

        pid = chunk[spec.profile_cols[0]].astype(str)
        for c in spec.profile_cols[1:]:
            pid = pid + "|" + chunk[c].astype(str)
        pid = pid + "|" + chunk["_date"].astype(str)
        chunk["_profile"] = pid

        last_prof = chunk["_profile"].iloc[-1]
        carry = chunk[chunk["_profile"] == last_prof].copy()
        chunk = chunk[chunk["_profile"] != last_prof]
        if len(chunk) == 0:
            continue

        for _, g in chunk.groupby("_profile", sort=False):
            yield g

    if carry is not None and len(carry):
        for _, g in carry.groupby("_profile", sort=False):
            yield g


def collect_cast_Tbins_and_class(g: pd.DataFrame, spec: SourceSpec) -> Tuple[np.ndarray, bool, Optional[str]]:
    z = pd.to_numeric(g.get(spec.depth_col), errors="coerce").to_numpy(dtype=float)
    mZ = np.isfinite(z)
    if spec.depth_qc_col:
        mZ = mZ & _qc_mask(g, spec.depth_qc_col, spec.good_qf)

    if not np.any(mZ):
        return np.zeros(NBINS, dtype=bool), False, None

    T = pd.to_numeric(g.get(spec.temp_col), errors="coerce").to_numpy(dtype=float)
    mT = mZ & np.isfinite(T) & (T >= -5.0) & (T <= 45.0)
    if spec.temp_qc_col:
        mT = mT & _qc_mask(g, spec.temp_qc_col, spec.good_qf)

    present = np.zeros(NBINS, dtype=bool)
    if np.any(mT):
        bi = bin_indices(z[mT])
        ok = (bi >= 0) & (bi < NBINS)
        if np.any(ok):
            present[np.unique(bi[ok])] = True

    cast_class = None
    if spec.name == "SDN" and spec.sdn_instr_col and spec.sdn_instr_col in g.columns:
        row_classes = [classify_sdn_row(x) for x in g[spec.sdn_instr_col].astype(str).tolist()]
        cast_class = resolve_sdn_cast_class(row_classes)

    return present, True, cast_class


def count_casts_by_bin(basin_list: List[str]) -> pd.DataFrame:
    rows = []

    for spec in SOURCES:
        for g in iter_profiles_csv(spec):
            basin_raw = g[spec.basin_col].iloc[0] if spec.basin_col in g.columns else ""
            basin = normalize_basin_value(basin_raw)
            basin = apply_basin_remap(basin, spec.basin_remap)

            if basin.strip().lower() == "out":
                continue

            if basin == "":
                continue
            if basin_list and basin not in basin_list:
                continue

            t0 = pd.to_datetime(g[spec.time_col].iloc[0], errors="coerce")
            if pd.isna(t0):
                continue
            y = int(t0.year)
            season = month_to_season(int(t0.month))
            if season is None:
                continue

            period = None
            for plab, (ymin, ymax) in PERIODS.items():
                if ymin <= y <= ymax:
                    period = plab
                    break
            if period is None:
                continue

            present_bins, ok, sdn_cast_class = collect_cast_Tbins_and_class(g, spec)
            if not ok:
                continue

            idx = np.where(present_bins)[0]
            if idx.size == 0:
                continue

            if spec.name == "SDN":
                # KEEP ONLY CTD/MBT-XBT/Bottle. Ignore everything else (incl sensors).
                if sdn_cast_class is None:
                    continue
                for k in idx:
                    rows.append((basin, season, period, float(BIN_CENTERS[k]), "SDN", sdn_cast_class, 1))
            else:
                # Pool all Adriatic files into one marker category
                src = "ARGO" if spec.name == "ARGO" else "ADRIATIC"
                for k in idx:
                    rows.append((basin, season, period, float(BIN_CENTERS[k]), src, "", 1))

    df = pd.DataFrame(rows, columns=["basin", "season", "period", "z_m", "source", "sdn_class", "n_casts"])
    if df.empty:
        return df

    df = (
        df.groupby(["basin", "season", "period", "z_m", "source", "sdn_class"], as_index=False)["n_casts"]
          .sum()
    )
    return df


# =============================================================================
# PLOTTING
# =============================================================================

def plot_one_basin(df: pd.DataFrame, basin: str) -> None:
    seasons = list(SEASONS.keys())
    periods = list(PERIODS.keys())

    dfb = df[(df["basin"] == basin) & (df["z_m"] <= PLOT_ZMAX)].copy()
    if dfb.empty:
        print(f"[WARN] No data for basin {basin}")
        return

    depths = np.array(sorted(dfb["z_m"].unique()), dtype=float)

    fig, axes = plt.subplots(len(seasons), len(periods), figsize=(12.5, 13.5), sharey=True)
    if axes.ndim == 1:
        axes = axes.reshape((len(seasons), len(periods)))

    handles = [
        Patch(facecolor=SDN_COLORS["SDN_CTD"], label="SDN CTD"),
        Patch(facecolor=SDN_COLORS["SDN_MBTXBT"], label="SDN MBT/XBT"),
        Patch(facecolor=SDN_COLORS["SDN_Bottle"], label="SDN Bottle"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ARGO_COLOR, markeredgecolor="white",
                   markersize=6, label="ARGO"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=ADRI_COLOR, markeredgecolor="white",
                   markersize=6, label="ADRIATIC"),
    ]

    for i, season in enumerate(seasons):
        for j, period in enumerate(periods):
            ax = axes[i, j]
            sub = dfb[(dfb["season"] == season) & (dfb["period"] == period)]

            sdn = sub[sub["source"] == "SDN"]
            left = np.zeros_like(depths, dtype=float)

            for cls in SDN_CLASSES:
                scls = sdn[sdn["sdn_class"] == cls]
                if scls.empty:
                    continue
                m = dict(zip(scls["z_m"].astype(float), scls["n_casts"].astype(float)))
                vals = np.array([m.get(z, 0.0) for z in depths], dtype=float)

                ax.barh(
                    depths, vals, left=left,
                    height=0.8 * np.diff(np.r_[0, depths]).mean() if depths.size > 1 else 15.0,
                    color=SDN_COLORS.get(cls, "#999999"),
                    edgecolor="none",
                    alpha=0.95,
                )
                left = left + vals

            ar = sub[sub["source"] == "ARGO"]
            if not ar.empty:
                m = dict(zip(ar["z_m"].astype(float), ar["n_casts"].astype(float)))
                x = np.array([m.get(z, np.nan) for z in depths], dtype=float)
                ax.scatter(x, depths, s=22, c=ARGO_COLOR, edgecolors="white", linewidths=0.4, zorder=5)

            ad = sub[sub["source"] == "ADRIATIC"]
            if not ad.empty:
                m = dict(zip(ad["z_m"].astype(float), ad["n_casts"].astype(float)))
                x = np.array([m.get(z, np.nan) for z in depths], dtype=float)
                ax.scatter(x, depths, s=26, c=ADRI_COLOR, marker="s", edgecolors="white", linewidths=0.4, zorder=5)

            ax.grid(True, alpha=0.25)
            ax.set_title(f"{season} | {period}", fontsize=11)
            if j == 0:
                ax.set_ylabel("Depth (m)", fontsize=11)
            if i == len(seasons) - 1:
                ax.set_xlabel("casts with ≥1 valid T sample in bin", fontsize=11)

            ax.set_ylim(PLOT_ZMAX, 0)

    fig.suptitle(
        f"{basin} — casts contributing to T bins (SDN stacked by platform; ARGO/ADRIATIC as markers)",
        fontsize=14, y=0.985
    )
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.965))
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out = OUTDIR / f"cast_counts_bybin_T_{basin}_rowsSeasons_colsPeriods.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("Saved:", out)


def main():
    basin_list = []  # empty -> no filtering during counting
    print("Counting casts-by-bin (T only)...")
    df = count_casts_by_bin(basin_list)
    
    #print(df[df["basin"].astype(str).str.lower().eq("out")]
      #.groupby(["source","period","season"], as_index=False)["n_casts"].sum()
      #.sort_values("n_casts", ascending=False)
      #.head(50))

    if df.empty:
        print("No data found. Check input paths.")
        return

    out_csv = "/Users/elena/Projects/IRB/MedClim/Stats/cast_counts_bybin_by_platform_season_period_T.csv"
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

    basins = sorted(df["basin"].unique().tolist())
    print("Basins detected:", basins)

    for b in basins:
        plot_one_basin(df, b)

    print("Done.")


if __name__ == "__main__":
    main()
