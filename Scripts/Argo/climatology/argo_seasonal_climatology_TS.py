#!/usr/bin/env python3
# Reads : Data/Argo/CSV/Argo_with_basins.csv
# Writes: Data/Argo/CSV/argo_manca_like_TS_pass1.csv, argo_manca_like_TS_after_1sigma.csv
"""
Compute seasonal (JFM/AMJ/JAS/OND) temperature & salinity climatological profiles per basin
from an Argo "points" CSV (one row per measurement level), in the same Manca-style workflow
as your SeaDataNet script:

1) PASS 1: QC + interpolate each profile to STD_DEPTHS, then compute mean/std across profiles
2) PASS 2: reject points where |x - mu| > 1 sigma (one iteration) and recompute

Output is a single CSV in the same schema expected by plot_manca_like_profiles_per_basin.py:
basin, season, z_m, n_T, T_mean, T_std, n_S, S_mean, S_std

Tested on the provided short.csv structure.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Scripts.paths import DATA

# -----------------------------
# INPUT COLUMNS (Argo export)
# -----------------------------
TIME_COL    = "TIME"
BASIN_COL   = "basin"
PLATFORM_COL= "PLATFORM_NUMBER"
CYCLE_COL   = "CYCLE_NUMBER"

PRES_COL    = "PRES"   # dbar ~ meters (good approximation for these plots)
TEMP_COL    = "TEMP"
SAL_COL     = "PSAL"

PRES_QC_COL = "PRES_QC"
TEMP_QC_COL = "TEMP_QC"
SAL_QC_COL  = "PSAL_QC"

# QC flags: Argo convention uses '1' (good) and '2' (probably good)
GOOD_QF_DEFAULT = "1,2"

SEASONS = {
    "JFM": {1, 2, 3},
    "AMJ": {4, 5, 6},
    "JAS": {7, 8, 9},
    "OND": {10, 11, 12},
}

STD_DEPTHS = np.array([
    0, 5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200,
    250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,
    1200, 1300, 1400, 1500, 1750, 2000, 2500, 3000,
    3500, 4000, 4500, 5000
], dtype=float)

# -----------------------------
# Interpolation utilities (same logic as SeaDataNet script)
# -----------------------------
def lagrange3(x, x1, y1, x2, y2, x3, y3):
    t1 = (x - x2) * (x - x3) / ((x1 - x2) * (x1 - x3))
    t2 = (x - x1) * (x - x3) / ((x2 - x1) * (x2 - x3))
    t3 = (x - x1) * (x - x2) / ((x3 - x1) * (x3 - x2))
    return y1 * t1 + y2 * t2 + y3 * t3

def reiniger_ross(z, v, zstd):
    """
    Weighted parabolic interpolation (Reiniger & Ross):
      - Use two 3-point parabolas around the bracket and weight between them.
      - Falls back near ends.
    Returns array of len(zstd) with NaNs outside observed range.
    """
    z = np.asarray(z, float)
    v = np.asarray(v, float)
    m = np.isfinite(z) & np.isfinite(v)
    z = z[m]; v = v[m]
    if z.size < 2:
        return np.full_like(zstd, np.nan, dtype=float)

    idx = np.argsort(z)
    z = z[idx]; v = v[idx]

    # compress duplicate pressures
    uz, inv, counts = np.unique(z, return_inverse=True, return_counts=True)
    if uz.size != z.size:
        vsum = np.zeros_like(uz, dtype=float)
        np.add.at(vsum, inv, v)
        v = vsum / counts
        z = uz

    out = np.full_like(zstd, np.nan, dtype=float)
    for j, zz in enumerate(zstd):
        if zz < z[0] or zz > z[-1]:
            continue
        k = np.searchsorted(z, zz)
        if k == 0:
            out[j] = v[0]
            continue
        if k >= len(z):
            out[j] = v[-1]
            continue

        # indices for left bracket (k-1) and right bracket (k)
        i2 = k - 1
        i3 = k

        # choose third point for left parabola
        i1 = i2 - 1 if i2 - 1 >= 0 else i2
        # choose third point for right parabola
        i4 = i3 + 1 if i3 + 1 < len(z) else i3

        # if we don't have enough unique points, fallback to linear
        if len({i1, i2, i3}) < 3 or len({i2, i3, i4}) < 3:
            out[j] = np.interp(zz, z, v)
            continue

        pL = lagrange3(zz, z[i1], v[i1], z[i2], v[i2], z[i3], v[i3])
        pR = lagrange3(zz, z[i2], v[i2], z[i3], v[i3], z[i4], v[i4])

        # weight based on proximity to bracket points
        denom = (z[i3] - z[i2])
        if denom == 0:
            out[j] = np.interp(zz, z, v)
        else:
            w = (zz - z[i2]) / denom
            out[j] = (1 - w) * pL + w * pR

    return out

def linear_interp_nan(z, v, zstd):
    z = np.asarray(z, float)
    v = np.asarray(v, float)
    zstd = np.asarray(zstd, float)

    out = np.full_like(zstd, np.nan, dtype=float)
    m = np.isfinite(z) & np.isfinite(v)
    z = z[m]; v = v[m]
    if z.size < 2:
        return out

    idx = np.argsort(z)
    z = z[idx]; v = v[idx]
    m2 = (zstd >= z[0]) & (zstd <= z[-1])
    if m2.any():
        out[m2] = np.interp(zstd[m2], z, v)
    return out

def sanitize_interp(vstd, var_kind):
    """Hard sanity for interpolated series to avoid blow-ups."""
    vstd = np.asarray(vstd, float)
    if var_kind == "TEMP":
        vstd[(vstd < -5.0) | (vstd > 45.0)] = np.nan
    elif var_kind == "SAL":
        vstd[(vstd < 0.0) | (vstd > 60.0)] = np.nan
    return vstd

# -----------------------------
# Accumulator
# -----------------------------
class Welford:
    __slots__ = ("n", "mean", "M2")
    def __init__(self, size):
        self.n = np.zeros(size, dtype=int)
        self.mean = np.zeros(size, dtype=float)
        self.M2 = np.zeros(size, dtype=float)

    def update_vec(self, x):
        x = np.asarray(x, float)
        m = np.isfinite(x)
        if not m.any():
            return
        xm = x[m]
        n0 = self.n[m].copy()
        self.n[m] += 1
        delta = xm - self.mean[m]
        self.mean[m] += delta / self.n[m]
        delta2 = xm - self.mean[m]
        self.M2[m] += delta * delta2

    def finalize(self):
        var = np.full_like(self.mean, np.nan, dtype=float)
        ok = self.n > 1
        var[ok] = self.M2[ok] / (self.n[ok] - 1)
        std = np.sqrt(var)
        mean = self.mean.copy()
        mean[self.n == 0] = np.nan
        return mean, std, self.n.copy()

# -----------------------------
# Reading and grouping into profiles
# -----------------------------
def iter_profiles(csv_path, chunksize=400_000, usecols=None):
    """
    Yield (profile_id, dataframe) where profile_id is PLATFORM|CYCLE|date.
    Assumes the CSV is ordered such that a profile's rows are contiguous (true for typical exports).
    """
    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        dtype=str,
        low_memory=False,
        usecols=(lambda c: True) if usecols is None else (lambda c: c in set(usecols)),
    )

    carry = None
    for chunk in reader:
        if carry is not None and len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)

        need = [BASIN_COL, TIME_COL, PLATFORM_COL, CYCLE_COL, PRES_COL]
        miss = [c for c in need if c not in chunk.columns]
        if miss:
            raise ValueError(f"Missing columns in chunk: {miss}")

        t = pd.to_datetime(chunk[TIME_COL], errors="coerce")
        chunk["_date"] = t.dt.date.astype(str)
        chunk["profile"] = (
            chunk[PLATFORM_COL].astype(str) + "|" +
            chunk[CYCLE_COL].astype(str) + "|" +
            chunk["_date"].astype(str)
        )

        last_prof = chunk["profile"].iloc[-1]
        carry = chunk[chunk["profile"] == last_prof].copy()
        chunk = chunk[chunk["profile"] != last_prof]
        if len(chunk) == 0:
            continue

        for pid, g in chunk.groupby("profile", sort=False):
            yield pid, g

    if carry is not None and len(carry):
        for pid, g in carry.groupby("profile", sort=False):
            yield pid, g

def collect_profile(g, var_col, var_qc_col=None, pres_qc_col=None, qc_on=True, good_qf=("1","2")):
    z = pd.to_numeric(g.get(PRES_COL), errors="coerce").to_numpy()
    v = pd.to_numeric(g.get(var_col), errors="coerce").to_numpy()
    m = np.isfinite(z) & np.isfinite(v)

    if qc_on:
        if var_qc_col and (var_qc_col in g.columns):
            q = g[var_qc_col].astype(str).str.strip().to_numpy()
            m = m & np.isin(q, list(good_qf))
        if pres_qc_col and (pres_qc_col in g.columns):
            qz = g[pres_qc_col].astype(str).str.strip().to_numpy()
            m = m & np.isin(qz, list(good_qf))

    # hard sanity
    m = m & (np.abs(v) < 1e4)

    if var_col == TEMP_COL:
        m = m & (v >= -5.0) & (v <= 45.0)
    elif var_col == SAL_COL:
        m = m & (v >= 0.0) & (v <= 60.0)

    # pressure/depth bounds
    m = m & (z >= 0.0) & (z <= 6000.0)

    z = z[m]; v = v[m]
    if z.size < 2:
        return None
    return z, v

def month_to_season(month):
    for s, months in SEASONS.items():
        if month in months:
            return s
    return None

def pass_stats(csv_path, year_min, year_max, qc_on=True, good_qf=("1","2"),
              first_pass_means=None, first_pass_stds=None,
              chunksize=400_000):
    accT, accS = {}, {}
    for profile_id, g in iter_profiles(
        csv_path,
        chunksize=chunksize,
        usecols=[
            BASIN_COL, TIME_COL, PLATFORM_COL, CYCLE_COL,
            PRES_COL, PRES_QC_COL,
            TEMP_COL, TEMP_QC_COL,
            SAL_COL, SAL_QC_COL
        ],
    ):
        basin = str(g[BASIN_COL].iloc[0]) if BASIN_COL in g.columns else "none"
        if basin in ("none", "nan", "NaN"):
            continue

        t0 = pd.to_datetime(g[TIME_COL].iloc[0], errors="coerce")
        if pd.isna(t0):
            continue
        yr = int(t0.year)
        if (yr < year_min) or (yr > year_max):
            continue
        season = month_to_season(int(t0.month))
        if season is None:
            continue

        keyT = (basin, season)
        keyS = (basin, season)

        # Temperature
        zv = collect_profile(g, TEMP_COL, TEMP_QC_COL, PRES_QC_COL, qc_on=qc_on, good_qf=good_qf)
        if zv is not None:
            z, v = zv
            vstd = linear_interp_nan(z, v, STD_DEPTHS) if len(z) < 5 else reiniger_ross(z, v, STD_DEPTHS)
            if np.any(np.isfinite(vstd)):
                vstd = sanitize_interp(vstd, "TEMP")

            # Pass-2 rejection
            if (first_pass_means is not None) and (first_pass_stds is not None):
                mu = first_pass_means.get(("T", basin, season), None)
                sd = first_pass_stds.get(("T", basin, season), None)
                if (mu is not None) and (sd is not None):
                    m = np.isfinite(vstd) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
                    vstd[m & (np.abs(vstd - mu) > sd)] = np.nan

            accT.setdefault(keyT, Welford(len(STD_DEPTHS))).update_vec(vstd)

        # Salinity
        zv = collect_profile(g, SAL_COL, SAL_QC_COL, PRES_QC_COL, qc_on=qc_on, good_qf=good_qf)
        if zv is not None:
            z, v = zv
            vstd = linear_interp_nan(z, v, STD_DEPTHS) if len(z) < 5 else reiniger_ross(z, v, STD_DEPTHS)
            if np.any(np.isfinite(vstd)):
                vstd = sanitize_interp(vstd, "SAL")

            if (first_pass_means is not None) and (first_pass_stds is not None):
                mu = first_pass_means.get(("S", basin, season), None)
                sd = first_pass_stds.get(("S", basin, season), None)
                if (mu is not None) and (sd is not None):
                    m = np.isfinite(vstd) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
                    vstd[m & (np.abs(vstd - mu) > sd)] = np.nan

            accS.setdefault(keyS, Welford(len(STD_DEPTHS))).update_vec(vstd)

    return accT, accS

def dump_csv(accT, accS, out_csv):
    rows = []
    keys = sorted(set(list(accT.keys()) + list(accS.keys())))
    for (basin, season) in keys:
        tmean, tstd, tn = (np.full(len(STD_DEPTHS), np.nan), np.full(len(STD_DEPTHS), np.nan), np.zeros(len(STD_DEPTHS), int))
        smean, sstd, sn = (np.full(len(STD_DEPTHS), np.nan), np.full(len(STD_DEPTHS), np.nan), np.zeros(len(STD_DEPTHS), int))

        if (basin, season) in accT:
            tmean, tstd, tn = accT[(basin, season)].finalize()
        if (basin, season) in accS:
            smean, sstd, sn = accS[(basin, season)].finalize()

        for k, z in enumerate(STD_DEPTHS):
            rows.append({
                "basin": basin,
                "season": season,
                "z_m": float(z),
                "n_T": int(tn[k]) if np.isfinite(tn[k]) else 0,
                "T_mean": float(tmean[k]) if np.isfinite(tmean[k]) else np.nan,
                "T_std": float(tstd[k]) if np.isfinite(tstd[k]) else np.nan,
                "n_S": int(sn[k]) if np.isfinite(sn[k]) else 0,
                "S_mean": float(smean[k]) if np.isfinite(smean[k]) else np.nan,
                "S_std": float(sstd[k]) if np.isfinite(sstd[k]) else np.nan,
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out

def main():
    infile = DATA / "Argo" / "CSV" / "Argo_with_basins.csv"
    out1   = DATA / "Argo" / "CSV" / "argo_manca_like_TS_pass1.csv"
    out2   = DATA / "Argo" / "CSV" / "argo_manca_like_TS_after_1sigma.csv"

    year_min  = 2000
    year_max  = 2026
    chunksize = 400_000

    qc_on   = True
    good_qf = ("1", "2")

    print("PASS 1: compute mean/std profiles per (basin, season) ...")
    accT1, accS1 = pass_stats(
        str(infile),
        year_min=year_min,
        year_max=year_max,
        qc_on=qc_on,
        good_qf=good_qf,
        chunksize=chunksize,
    )
    dump_csv(accT1, accS1, str(out1))
    print("Wrote:", out1)

    # build mean/std dicts for pass-2 rejection
    means, stds = {}, {}
    for (basin, season), w in accT1.items():
        mu, sd, _ = w.finalize()
        means[("T", basin, season)] = mu
        stds[("T", basin, season)] = sd
    for (basin, season), w in accS1.items():
        mu, sd, _ = w.finalize()
        means[("S", basin, season)] = mu
        stds[("S", basin, season)] = sd

    print("PASS 2: reject |x-mu|>1σ and recompute (one iteration, Manca-style) ...")
    accT2, accS2 = pass_stats(
        str(infile),
        year_min=year_min,
        year_max=year_max,
        qc_on=qc_on,
        good_qf=good_qf,
        first_pass_means=means,
        first_pass_stds=stds,
        chunksize=chunksize,
    )
    dump_csv(accT2, accS2, str(out2))
    print("Wrote:", out2)

if __name__ == "__main__":
    main()
