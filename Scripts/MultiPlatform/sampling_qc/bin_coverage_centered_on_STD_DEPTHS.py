#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STD-centered bin coverage diagnostics with VARIABLE-SPECIFIC OK bins (T vs S).

What it measures (per source/period/basin/season and per bin):
  - n_profiles_total
  - frac_reach: fraction of profiles whose zmax >= bin_center
  - frac_T_in_bin: fraction of profiles with >=1 valid T sample inside bin
  - frac_S_in_bin: fraction of profiles with >=1 valid S sample inside bin

Then it derives variable-specific "OK bin depth":
  - deepest bin for which frac_VAR_in_bin >= MIN_FRAC_VAR and n_VAR_profiles_in_bin >= MIN_N_VAR
    (separately for T and for S)

Outputs:
  1) bin_coverage_<TAG>_varOK.csv
  2) flags_bins_<TAG>_varOK.csv
  3) max_depth_ok_<TAG>_varOK.csv
  4) coverage_fraction_<TAG>_varOK.png

IMPORTANT:
  - This does NOT interpolate. It just checks how much original data falls into each centered bin.
  - Coverage is variable-specific: a profile can contribute to T but not to S (or vice versa).
  - "Reach" is purely geometric: zmax >= bin center (independent of variable availability).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

TAG = "ALL_SOURCES_STD_CENTERED"
OUTDIR = Path("/Users/elena/Projects/IRB/MedClim/Stats/Vertical_Sampling")
OUTDIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 400_000

PERIODS = {
    "20c_1950_1999": (1950, 1999),
    "21c_2000_2019": (2000, 2019),
}

SEASONS = {
    "JFM": {1, 2, 3},
    "AMJ": {4, 5, 6},
    "JAS": {7, 8, 9},
    "OND": {10, 11, 12},
}

# --- Standard depth centers (edit if needed) ---
STD_CENTERS = np.array([
    5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200,
    250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,
    1200, 1300, 1400, 1500, 1750, 2000, 2500, 3000,
    3500, 4000, 4500, 5000
], dtype=float)

# --- Variable-specific OK-bin thresholds ---
# Tune these to your desired strictness:
MIN_FRAC_T = 0.10   # e.g. "at least 10% of profiles have T in that bin"
MIN_FRAC_S = 0.05   # salinity often sparser -> allow lower fraction
MIN_N_T    = 30     # absolute minimum #profiles contributing T in bin
MIN_N_S    = 30     # absolute minimum #profiles contributing S in bin

# Depth sanity (meters; ARGO pressure is treated approximately as meters here)
ZMIN, ZMAX = 0.0, 6000.0

# Basin normalization for Adriatic 3-basin labels (if needed)
BASIN_ALIASES = {
    "nadr": "DJ1",
    "madr": "DJ2",
    "sadr": "DJ4",
    "north": "DJ1",
    "mid": "DJ2",
    "middle": "DJ2",
    "central": "DJ2",
    "south": "DJ4",
    "n": "DJ1",
    "m": "DJ2",
    "s": "DJ4",
}

def normalize_basin(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "nan"):
        return ""
    key = s.lower()
    out = BASIN_ALIASES.get(key, s)
    return str(out).strip().upper()

def month_to_season(m: int) -> Optional[str]:
    for s, months in SEASONS.items():
        if m in months:
            return s
    return None


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
    sal_col: str
    profile_cols: List[str]   # used to build profile id
    # Optional QC columns (if not present in the file, they’re ignored)
    temp_qc_col: Optional[str] = None
    sal_qc_col: Optional[str] = None
    good_qf: Optional[set] = None
    depth_kind: str = "depth_m"  # "depth_m" or "pres_dbar"

GOOD_QF_SDN  = {"1", "2"}
GOOD_QF_ARGO = {"1", "2"}

SOURCES: List[SourceSpec] = [
    SourceSpec(
        name="SDN",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins_mainland_special_excluded.csv"),
        time_col="yyyy-mm-ddThh:mm:ss.sss",
        basin_col="basin",
        depth_col="Depth [m]",
        temp_col="ITS-90 water temperature [degrees C]",
        sal_col="Water body salinity [per mille]",
        profile_cols=["Cruise", "Station:METAVAR:INDEXED_TEXT"],
        temp_qc_col="QV:SEADATANET.1",   # will be ignored if not present
        sal_qc_col="QV:SEADATANET.2",
        good_qf=GOOD_QF_SDN,
        depth_kind="depth_m",
    ),
    SourceSpec(
        name="ARGO",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Argo/CSV/Argo_with_basins_mainland_special_excluded.csv"),
        time_col="TIME",
        basin_col="basin",
        depth_col="PRES",   # dbar ~ meters (good enough for bin coverage)
        temp_col="TEMP",
        sal_col="PSAL",
        profile_cols=["PLATFORM_NUMBER", "CYCLE_NUMBER"],
        temp_qc_col="TEMP_QC",
        sal_qc_col="PSAL_QC",
        good_qf=GOOD_QF_ARGO,
        depth_kind="pres_dbar",
    ),
    SourceSpec(
        name="CTD_HR_2019_2023_MFSD",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/CTD_HR_2019_2023_MFSD_long_fixed_mainland_special_excluded.csv"),
        time_col="TIME",
        basin_col="BASIN_3ADR",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        profile_cols=["STATION"],
        depth_kind="depth_m",
    ),
    SourceSpec(
        name="RB_Northern_Adriatic_Transect_1979_2018",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/RB_Northern_Adriatic_Transect_1979_2018_long_fixed_mainland_special_excluded.csv"),
        time_col="TIME",
        basin_col="BASIN_3ADR",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        profile_cols=["STATION"],
        depth_kind="depth_m",
    ),
    SourceSpec(
        name="Palagruza_Transect_1952_2018",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/Palagruza_Transect_1952_2018_long_fixed_mainland_special_excluded.csv"),
        time_col="TIME",
        basin_col="BASIN_3ADR",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        profile_cols=["STATION"],
        depth_kind="depth_m",
    ),
    SourceSpec(
        name="IOR_Adriatic_database_1987_2019",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/IOR_Adriatic_database_1987_2019_long_fixed_mainland_special_excluded.csv"),
        time_col="TIME",
        basin_col="BASIN_3ADR",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        profile_cols=["STATION", "CAST_IDX"],
        depth_kind="depth_m",
    ),
]


# =============================================================================
# BINS (centered to STD_CENTERS; contiguous edges)
# =============================================================================

def make_centered_bin_edges(centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given bin centers (monotonic increasing), build contiguous bin edges as midpoints:
      edges[i]   = midpoint(centers[i-1], centers[i])
      first edge = max(0, centers[0] - 0.5*(centers[1]-centers[0])) (clamped)
      last edge  = centers[-1] + 0.5*(centers[-1]-centers[-2])

    Returns:
      centers, edges (len(edges)=len(centers)+1)
    """
    c = np.asarray(centers, float).copy()
    c.sort()
    if len(c) < 2:
        raise ValueError("Need at least 2 centers to build edges.")

    mid = 0.5 * (c[:-1] + c[1:])
    first = 0. #max(0.0, c[0] - 0.5 * (c[1] - c[0]))
    last  = c[-1] + 0.5 * (c[-1] - c[-2])

    edges = np.concatenate([[first], mid, [last]])
    return c, edges

BIN_CENTERS, BIN_EDGES = make_centered_bin_edges(STD_CENTERS)


# =============================================================================
# STREAMING PROFILE ITERATOR
# =============================================================================

def iter_profiles_csv(spec: SourceSpec, chunksize: int = CHUNKSIZE) -> Iterable[pd.DataFrame]:
    if not spec.path.exists():
        raise FileNotFoundError(f"[{spec.name}] Missing input: {spec.path}")

    reader = pd.read_csv(spec.path, chunksize=chunksize, dtype=str, low_memory=False)
    carry = None

    for chunk in reader:
        if carry is not None and len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)

        t = pd.to_datetime(chunk.get(spec.time_col), errors="coerce")
        chunk["_date"] = t.dt.date.astype(str)

        parts = []
        for c in spec.profile_cols:
            if c not in chunk.columns:
                raise ValueError(f"[{spec.name}] Missing profile col '{c}'")
            parts.append(chunk[c].astype(str))
        parts.append(chunk["_date"].astype(str))

        pid = parts[0]
        for p in parts[1:]:
            pid = pid + "|" + p
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


# =============================================================================
# CORE: per-profile bin membership (T and S separately)
# =============================================================================

def _qc_mask(g: pd.DataFrame, qc_col: Optional[str], good_qf: Optional[set]) -> np.ndarray:
    if qc_col is None or good_qf is None or qc_col not in g.columns:
        return np.ones(len(g), dtype=bool)
    q = g[qc_col].astype(str).str.strip().to_numpy()
    return np.isin(q, list(good_qf))

def _valid_depth(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    return np.isfinite(z) & (z >= ZMIN) & (z <= ZMAX)

def _valid_temp(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, float)
    return np.isfinite(T) & (np.abs(T) < 1e4) & (T >= -5.0) & (T <= 45.0)

def _valid_sal(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, float)
    return np.isfinite(S) & (np.abs(S) < 1e4) & (S >= 0.0) & (S <= 60.0)

def bins_touched(z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Return boolean vector len(nbins) telling which bins contain at least one z sample.
    """
    z = z[np.isfinite(z)]
    if z.size == 0:
        return np.zeros(len(edges) - 1, dtype=bool)
    # bin index for each z
    # np.searchsorted gives index in edges; subtract 1 -> bin id
    bi = np.searchsorted(edges, z, side="right") - 1
    ok = (bi >= 0) & (bi < (len(edges) - 1))
    bi = bi[ok]
    out = np.zeros(len(edges) - 1, dtype=bool)
    out[np.unique(bi)] = True
    return out


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_bin_coverage() -> pd.DataFrame:
    rows = []

    for spec in SOURCES:
        print(f"\n[{spec.name}] reading: {spec.path}")

        for g in iter_profiles_csv(spec, CHUNKSIZE):
            # basin
            basin_raw = g[spec.basin_col].iloc[0] if spec.basin_col in g.columns else ""
            basin = normalize_basin(basin_raw)
            if basin == "":
                continue

            # time -> period + season
            t0 = pd.to_datetime(g[spec.time_col].iloc[0], errors="coerce")
            if pd.isna(t0):
                continue
            year = int(t0.year)
            season = month_to_season(int(t0.month))
            if season is None:
                continue

            # period label
            period = None
            for lab, (ymin, ymax) in PERIODS.items():
                if ymin <= year <= ymax:
                    period = lab
                    break
            if period is None:
                continue

            # depth
            z = pd.to_numeric(g.get(spec.depth_col), errors="coerce").to_numpy()
            mZ = _valid_depth(z)

            z_ok = z[mZ]
            if z_ok.size == 0:
                continue
            zmax = float(np.nanmax(z_ok))

            # Temperature variable-specific validity
            T = pd.to_numeric(g.get(spec.temp_col), errors="coerce").to_numpy()
            mT = mZ & _valid_temp(T) & _qc_mask(g, spec.temp_qc_col, spec.good_qf)
            zT = z[mT]

            # Salinity variable-specific validity
            S = pd.to_numeric(g.get(spec.sal_col), errors="coerce").to_numpy()
            mS = mZ & _valid_sal(S) & _qc_mask(g, spec.sal_qc_col, spec.good_qf)
            zS = z[mS]

            touched_T = bins_touched(zT, BIN_EDGES) if zT.size else np.zeros(len(BIN_CENTERS), dtype=bool)
            touched_S = bins_touched(zS, BIN_EDGES) if zS.size else np.zeros(len(BIN_CENTERS), dtype=bool)

            rows.append({
                "source": spec.name,
                "period": period,
                "basin": basin,
                "season": season,
                "zmax": zmax,
                # store compactly: later we will explode to per-bin counts
                "touched_T": touched_T,
                "touched_S": touched_S,
            })

    prof = pd.DataFrame(rows)
    if prof.empty:
        raise RuntimeError("No profiles were collected. Check paths/columns/periods.")
    return prof


def summarize_to_bins(prof: pd.DataFrame) -> pd.DataFrame:
    """
    Explode per-profile touched arrays into per-bin group summary.
    """
    group_cols = ["source", "period", "basin", "season"]

    # Precompute reach matrix: for each profile, which bin centers are reached by zmax
    zmax = prof["zmax"].to_numpy(float)
    reach = (zmax[:, None] >= BIN_CENTERS[None, :])

    # Stack touched arrays
    touched_T = np.vstack(prof["touched_T"].to_numpy())
    touched_S = np.vstack(prof["touched_S"].to_numpy())

    # Build a working frame to group with integer profile index
    base = prof[group_cols].copy()
    base["_i"] = np.arange(len(base))

    out_rows = []
    for key, idx in base.groupby(group_cols, sort=False)["_i"]:
        idx = idx.to_numpy()
        n_total = int(len(idx))

        # counts per bin
        n_reach = reach[idx].sum(axis=0).astype(int)
        n_T     = touched_T[idx].sum(axis=0).astype(int)
        n_S     = touched_S[idx].sum(axis=0).astype(int)

        frac_reach = n_reach / n_total if n_total else np.full_like(n_reach, np.nan, dtype=float)
        frac_T     = n_T / n_total if n_total else np.full_like(n_T, np.nan, dtype=float)
        frac_S     = n_S / n_total if n_total else np.full_like(n_S, np.nan, dtype=float)

        for j, zc in enumerate(BIN_CENTERS):
            out_rows.append({
                "source": key[0],
                "period": key[1],
                "basin": key[2],
                "season": key[3],
                "bin_center_m": float(zc),
                "bin_lo_m": float(BIN_EDGES[j]),
                "bin_hi_m": float(BIN_EDGES[j+1]),
                "n_profiles_total": n_total,
                "n_reach": int(n_reach[j]),
                "frac_reach": float(frac_reach[j]),
                "n_T_profiles_in_bin": int(n_T[j]),
                "frac_T_in_bin": float(frac_T[j]),
                "n_S_profiles_in_bin": int(n_S[j]),
                "frac_S_in_bin": float(frac_S[j]),
            })

    return pd.DataFrame(out_rows)


def deepest_ok_bin(df_bins: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each group (source, period, basin, season), find deepest OK bin for T and for S.
    """
    group_cols = ["source", "period", "basin", "season"]

    rows_flags = []
    rows_depth = []

    for key, g in df_bins.groupby(group_cols, sort=False):
        g = g.sort_values("bin_center_m")
        zc = g["bin_center_m"].to_numpy(float)

        okT = (g["frac_T_in_bin"].to_numpy(float) >= MIN_FRAC_T) & (g["n_T_profiles_in_bin"].to_numpy(int) >= MIN_N_T)
        okS = (g["frac_S_in_bin"].to_numpy(float) >= MIN_FRAC_S) & (g["n_S_profiles_in_bin"].to_numpy(int) >= MIN_N_S)

        z_ok_T = float(np.nanmax(zc[okT])) if np.any(okT) else np.nan
        z_ok_S = float(np.nanmax(zc[okS])) if np.any(okS) else np.nan

        rows_flags.append({
            **dict(zip(group_cols, key)),
            "MIN_FRAC_T": MIN_FRAC_T,
            "MIN_FRAC_S": MIN_FRAC_S,
            "MIN_N_T": MIN_N_T,
            "MIN_N_S": MIN_N_S,
            "deepest_ok_bin_T_m": z_ok_T,
            "deepest_ok_bin_S_m": z_ok_S,
        })
        rows_depth.append({
            **dict(zip(group_cols, key)),
            "zmax_ok_T_m": z_ok_T,
            "zmax_ok_S_m": z_ok_S,
        })

    return pd.DataFrame(rows_flags), pd.DataFrame(rows_depth)


def plot_coverage(df_bins: pd.DataFrame, out_png: Path, trim_quantile: float = 0.99):
    """
    Clearer plot:
      - one panel per (source, period)
      - plots mean frac_T_in_bin and mean frac_S_in_bin over basin+season
      - AUTO-TRIMS x-axis to a robust "data depth limit" based on reach:
          depth_limit = max bin_center where mean(frac_reach) >= (1-trim_quantile)
        (equivalently: uses distribution of zmax implicitly via reach)
    """
    # mean across basin/season (unweighted; change if you want weighted by n_profiles_total)
    agg = (df_bins
           .groupby(["source", "period", "bin_center_m"], as_index=False)
           .agg(frac_reach=("frac_reach", "mean"),
                frac_T=("frac_T_in_bin", "mean"),
                frac_S=("frac_S_in_bin", "mean")))

    combos = agg[["source", "period"]].drop_duplicates().sort_values(["source", "period"])
    n = len(combos)
    if n == 0:
        return

    fig_h = max(6, 2.4 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, fig_h), sharex=False)
    if n == 1:
        axes = [axes]

    # This threshold means: show bins that are reached by at least (1-trim_quantile) fraction of profiles.
    # Example trim_quantile=0.99 -> threshold=0.01 (i.e., at least 1% of profiles reach the bin).
    reach_thr = max(0.0, 1.0 - float(trim_quantile))

    for ax, (_, row) in zip(axes, combos.iterrows()):
        src = row["source"]; per = row["period"]
        g = (agg[(agg["source"] == src) & (agg["period"] == per)]
             .sort_values("bin_center_m")
             .reset_index(drop=True))

        # --- Determine a robust x-limit ---
        # deepest bin where mean frac_reach >= reach_thr
        if np.any(g["frac_reach"].to_numpy(float) >= reach_thr):
            depth_limit = float(g.loc[g["frac_reach"] >= reach_thr, "bin_center_m"].max())
        else:
            # fallback: just show the first few bins if nothing reaches anything (shouldn't happen)
            depth_limit = float(g["bin_center_m"].iloc[min(8, len(g)-1)])

        g_plot = g[g["bin_center_m"] <= depth_limit].copy()
        if len(g_plot) < 3:
            # make sure we show at least a few points
            g_plot = g.iloc[:min(10, len(g))].copy()
            depth_limit = float(g_plot["bin_center_m"].max())

        ax.plot(g_plot["bin_center_m"], g_plot["frac_T"], marker="o", linewidth=1.2, label="T in-bin fraction")
        ax.plot(g_plot["bin_center_m"], g_plot["frac_S"], marker="o", linewidth=1.2, label="S in-bin fraction")

        # optional: show reach as a faint dashed line (helps interpret why axis is truncated)
        ax.plot(g_plot["bin_center_m"], g_plot["frac_reach"], linestyle="--", linewidth=1.0, label="Reach fraction")

        ax.set_title(f"{src} | {per}  (trim at ~{int(trim_quantile*100)}th pct reach)")
        ax.set_ylabel("Fraction of profiles")
        ax.set_xlim(0, depth_limit)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Bin center depth (m)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)



def main():
    out_bins   = OUTDIR / f"bin_coverage_{TAG}_varOK.csv"
    out_flags  = OUTDIR / f"flags_bins_{TAG}_varOK.csv"
    out_depth  = OUTDIR / f"max_depth_ok_{TAG}_varOK.csv"
    out_png    = OUTDIR / f"coverage_fraction_{TAG}_varOK.png"

    print("Computing per-profile bin touches (T & S separately) ...")
    prof = compute_bin_coverage()

    print("Summarizing to per-bin coverage table ...")
    df_bins = summarize_to_bins(prof)
    df_bins.to_csv(out_bins, index=False)
    print("Wrote:", out_bins)

    print("Computing variable-specific deepest OK bins ...")
    df_flags, df_depth = deepest_ok_bin(df_bins)
    df_flags.to_csv(out_flags, index=False)
    df_depth.to_csv(out_depth, index=False)
    print("Wrote:", out_flags)
    print("Wrote:", out_depth)

    print("Plotting coverage fractions (mean over basins+seasons) ...")
    plot_coverage(df_bins, out_png)
    print("Wrote:", out_png)

    print("\nDone.")


if __name__ == "__main__":
    main()
