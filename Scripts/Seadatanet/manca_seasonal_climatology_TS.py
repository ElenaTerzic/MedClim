#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path

# ---- INPUT COLUMNS (adjust if needed) ----
DATE_COL    = "yyyy-mm-ddThh:mm:ss.sss"
BASIN_COL   = "basin"
CRUISE_COL  = "Cruise"
STATION_COL = "Station:METAVAR:INDEXED_TEXT"
LON_COL     = "Longitude [degrees_east]"
LAT_COL     = "Latitude [degrees_north]"

CAST_COL    = "LOCAL_CDI_ID"  # IMPORTANT: define profiles by this
DEPTH_COL   = "Depth [m]"
TEMP_COL    = "ITS-90 water temperature [degrees C]"
SAL_COL     = "Water body salinity [per mille]"

# Default; overridden in main() for multi-period runs
YEAR_MIN, YEAR_MAX = 1950, 1999

# Period definitions for multi-run output
PERIODS = {
    "20c_1950_1999": (1950, 1999),
    "21c_2000_2019": (2000, 2019),
}

# QC columns
TEMP_QC_COL = "QV:SEADATANET.1"
SAL_QC_COL  = "QV:SEADATANET.2"
DEPTH_QC_COL = "QV:SEADATANET"  # optional; often present

GOOD_QF_NUM = {1, 2}  # treat QC as numeric

DEBUG_PRINT_LIMIT = 200

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

CHUNKSIZE = 400_000

# -------- Reiniger & Ross-like weighted parabolic interpolation --------
def _parab3(x1, y1, x2, y2, x3, y3, x):
    t1 = (x - x2) * (x - x3) / ((x1 - x2) * (x1 - x3))
    t2 = (x - x1) * (x - x3) / ((x2 - x1) * (x2 - x3))
    t3 = (x - x1) * (x - x2) / ((x3 - x1) * (x3 - x2))
    return y1 * t1 + y2 * t2 + y3 * t3

def reiniger_ross(z, v, zstd):
    z = np.asarray(z, float)
    v = np.asarray(v, float)
    m = np.isfinite(z) & np.isfinite(v)
    z = z[m]; v = v[m]
    if z.size < 2:
        return np.full_like(zstd, np.nan, dtype=float)

    idx = np.argsort(z)
    z = z[idx]; v = v[idx]

    uz, inv, counts = np.unique(z, return_inverse=True, return_counts=True)
    if uz.size != z.size:
        vs = np.zeros_like(uz, dtype=float)
        np.add.at(vs, inv, v)
        v = vs / counts
        z = uz

    out = np.full_like(zstd, np.nan, dtype=float)
    n = z.size

    for j, zz in enumerate(zstd):
        if zz < z[0] or zz > z[-1]:
            continue

        if n == 2:
            out[j] = np.interp(zz, z, v)
            continue

        k = np.searchsorted(z, zz) - 1
        if k < 0:
            k = 0
        if k >= n - 1:
            k = n - 2

        if k == 0:
            out[j] = _parab3(z[0], v[0], z[1], v[1], z[2], v[2], zz)
            continue
        if k >= n - 2:
            out[j] = _parab3(z[n-3], v[n-3], z[n-2], v[n-2], z[n-1], v[n-1], zz)
            continue

        p1 = _parab3(z[k-1], v[k-1], z[k], v[k], z[k+1], v[k+1], zz)
        p2 = _parab3(z[k], v[k], z[k+1], v[k+1], z[k+2], v[k+2], zz)
        w = (zz - z[k]) / (z[k+1] - z[k]) if z[k+1] != z[k] else 0.0
        out[j] = (1.0 - w) * p1 + w * p2

    return out

def clamp_to_bracket(z, v, zstd, vstd):
    z = np.asarray(z, float)
    v = np.asarray(v, float)
    zstd = np.asarray(zstd, float)
    vstd = np.asarray(vstd, float)

    m = np.isfinite(z) & np.isfinite(v)
    z = z[m]; v = v[m]
    if z.size < 2:
        return np.full_like(zstd, np.nan, dtype=float)

    # sort
    order = np.argsort(z)
    z = z[order]; v = v[order]

    # compress duplicate depths (average)
    uz, inv, counts = np.unique(z, return_inverse=True, return_counts=True)
    if uz.size != z.size:
        vs = np.zeros_like(uz, dtype=float)
        np.add.at(vs, inv, v)
        v = vs / counts
        z = uz

    out = np.asarray(vstd, float).copy()
    n = len(z)

    for j, zz in enumerate(zstd):
        if not np.isfinite(out[j]):
            continue
        if zz < z[0] or zz > z[-1]:
            out[j] = np.nan
            continue

        k = np.searchsorted(z, zz) - 1
        if k < 0:
            k = 0
        if k >= n - 1:
            k = n - 2

        lo = min(v[k], v[k+1])
        hi = max(v[k], v[k+1])
        out[j] = min(max(out[j], lo), hi)

    return out


def linear_interp_nan(z, v, zstd):
    z = np.asarray(z, float)
    v = np.asarray(v, float)
    zstd = np.asarray(zstd, float)

    out = np.full_like(zstd, np.nan, dtype=float)
    if z.size < 2:
        return out

    order = np.argsort(z)
    z = z[order]
    v = v[order]

    m = (zstd >= z[0]) & (zstd <= z[-1])
    if np.any(m):
        out[m] = np.interp(zstd[m], z, v)
    return out

def sanitize_interp(vstd, var_col):
    vstd = np.asarray(vstd, float)

    vstd[np.abs(vstd) > 1e4] = np.nan

    if var_col == TEMP_COL:
        vstd[(vstd < -5.0) | (vstd > 45.0)] = np.nan
    elif var_col == SAL_COL:
        vstd[(vstd < 0.0) | (vstd > 60.0)] = np.nan

    return vstd

# -------- Streaming cast iterator (LOCAL_CDI_ID first; chunk carryover) --------
def month_to_season(m):
    for s, months in SEASONS.items():
        if m in months:
            return s
    return None

def _build_cast_id(chunk: pd.DataFrame) -> pd.Series:
    """
    Use LOCAL_CDI_ID when present; else fallback to Cruise|Station|date.
    """
    cid = chunk.get(CAST_COL)
    if cid is None:
        cid = pd.Series([""] * len(chunk), index=chunk.index)

    cid = cid.astype(str).str.strip()
    has = cid.ne("") & cid.ne("nan") & cid.ne("None")

    t = pd.to_datetime(chunk[DATE_COL], errors="coerce")
    date = t.dt.strftime("%Y-%m-%d").fillna("")

    fb = (
        chunk[CRUISE_COL].astype(str).fillna("") + "|" +
        chunk[STATION_COL].astype(str).fillna("") + "|" +
        date
    )

    return cid.where(has, fb)

def iter_casts(csv_path, chunksize=CHUNKSIZE):
    usecols = [
        BASIN_COL, DATE_COL, CRUISE_COL, STATION_COL, CAST_COL,
        LON_COL, LAT_COL, DEPTH_COL,
        TEMP_COL, TEMP_QC_COL,
        SAL_COL, SAL_QC_COL
    ]
    # optional depth qc
    opt = [DEPTH_QC_COL]
    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        dtype=str,
        low_memory=False,
        usecols=lambda c: (c in set(usecols)) or (c in set(opt))
    )

    carry = None

    for chunk in reader:
        if carry is not None and len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)

        need = [BASIN_COL, DATE_COL, DEPTH_COL]
        miss = [c for c in need if c not in chunk.columns]
        if miss:
            raise ValueError(f"Missing columns in chunk: {miss}")

        chunk["cast"] = _build_cast_id(chunk)

        last_cast = chunk["cast"].iloc[-1]
        carry = chunk[chunk["cast"] == last_cast].copy()
        chunk = chunk[chunk["cast"] != last_cast]
        if len(chunk) == 0:
            continue

        for cast_id, g in chunk.groupby("cast", sort=False):
            yield cast_id, g

    if carry is not None and len(carry):
        for cast_id, g in carry.groupby("cast", sort=False):
            yield cast_id, g

# -------- Welford accumulator per (basin, season, depth_index) --------
class Welford:
    __slots__ = ("n", "mean", "M2")
    def __init__(self, size):
        self.n = np.zeros(size, dtype=np.int64)
        self.mean = np.zeros(size, dtype=float)
        self.M2 = np.zeros(size, dtype=float)

    def update_vec(self, x):
        m = np.isfinite(x)
        if not np.any(m):
            return
        xm = x[m]
        n_old = self.n[m]
        self.n[m] = n_old + 1
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

def collect_profile(g, var_col, qc_col=None, qc_on=True, enforce_depth_qc=False):
    z = pd.to_numeric(g.get(DEPTH_COL), errors="coerce").to_numpy()
    v = pd.to_numeric(g.get(var_col), errors="coerce").to_numpy()

    m = np.isfinite(z) & np.isfinite(v)

    # OPTIONAL: enforce depth QC if present
    if enforce_depth_qc and (DEPTH_QC_COL in g.columns):
        qd = pd.to_numeric(g[DEPTH_QC_COL], errors="coerce").to_numpy()
        m = m & np.isin(qd, list(GOOD_QF_NUM))

    # Variable QC (numeric)
    if qc_on and (qc_col in g.columns):
        q = pd.to_numeric(g[qc_col], errors="coerce").to_numpy()
        m = m & np.isin(q, list(GOOD_QF_NUM))

    # hard sanity
    m = m & (np.abs(v) < 1e4)

    if var_col == TEMP_COL:
        m = m & (v >= -5.0) & (v <= 45.0)
    elif var_col == SAL_COL:
        m = m & (v >= 0.0) & (v <= 60.0)

    m = m & (z >= 0.0) & (z <= 6000.0)

    z = z[m]; v = v[m]
    if z.size < 2:
        return None
    return z, v

def pass_stats(csv_path, qc_on=True, first_pass_means=None, first_pass_stds=None,
               chunksize=CHUNKSIZE, enforce_depth_qc=False):
    accT = {}
    accS = {}

    dbg_temp = 0
    dbg_sal  = 0

    for cast_id, g in iter_casts(csv_path, chunksize=chunksize):
        basin = str(g[BASIN_COL].iloc[0]) if BASIN_COL in g.columns else "none"
        if basin in ("none", "nan", "NaN"):
            continue

        t0 = pd.to_datetime(g[DATE_COL].iloc[0], errors="coerce")
        if pd.isna(t0):
            continue

        y = int(t0.year)
        if y < YEAR_MIN or y > YEAR_MAX:
            continue

        season = month_to_season(int(t0.month))
        if season is None:
            continue

        key = (basin, season)

        # Temperature
        zv = collect_profile(g, TEMP_COL, TEMP_QC_COL, qc_on=qc_on, enforce_depth_qc=enforce_depth_qc)
        if zv is not None:
            z, v = zv

            if len(z) < 5:
                vstd = linear_interp_nan(z, v, STD_DEPTHS)
            else:
                vstd = reiniger_ross(z, v, STD_DEPTHS)
                # debug RR overshoot BEFORE clamping (guard all-nan)
                if np.isfinite(vstd).any():
                    mn = np.nanmin(vstd); mx = np.nanmax(vstd)
                    if (mn < -5) or (mx > 45):
                        if dbg_temp < DEBUG_PRINT_LIMIT:
                            print("RR_OVERSHOOT_TEMP", key, cast_id, "nlev", len(z),
                                  "min/max", mn, mx)
                        dbg_temp += 1

                vstd = clamp_to_bracket(z, v, STD_DEPTHS, vstd)

            vstd = sanitize_interp(vstd, TEMP_COL)

            if first_pass_means is not None:
                mu = first_pass_means[("T",) + key]
                sd = first_pass_stds[("T",) + key]
                ok = np.isfinite(vstd) & np.isfinite(mu) & np.isfinite(sd) & (np.abs(vstd - mu) <= sd)
                vstd = np.where(ok, vstd, np.nan)
                vstd = sanitize_interp(vstd, TEMP_COL)

            accT.setdefault(key, Welford(len(STD_DEPTHS))).update_vec(vstd)

        # Salinity
        zv = collect_profile(g, SAL_COL, SAL_QC_COL, qc_on=qc_on, enforce_depth_qc=enforce_depth_qc)
        if zv is not None:
            z, v = zv

            if len(z) < 5:
                vstd = linear_interp_nan(z, v, STD_DEPTHS)
            else:
                vstd = reiniger_ross(z, v, STD_DEPTHS)
                if np.isfinite(vstd).any():
                    mn = np.nanmin(vstd); mx = np.nanmax(vstd)
                    if (mn < 0) or (mx > 60):
                        if dbg_sal < DEBUG_PRINT_LIMIT:
                            print("RR_OVERSHOOT_SAL", key, cast_id, "nlev", len(z),
                                  "min/max", mn, mx)
                        dbg_sal += 1

                vstd = clamp_to_bracket(z, v, STD_DEPTHS, vstd)

            vstd = sanitize_interp(vstd, SAL_COL)

            if first_pass_means is not None:
                mu = first_pass_means[("S",) + key]
                sd = first_pass_stds[("S",) + key]
                ok = np.isfinite(vstd) & np.isfinite(mu) & np.isfinite(sd) & (np.abs(vstd - mu) <= sd)
                vstd = np.where(ok, vstd, np.nan)
                vstd = sanitize_interp(vstd, SAL_COL)

            accS.setdefault(key, Welford(len(STD_DEPTHS))).update_vec(vstd)

    # one summary per pass (not per cast)
    print("Total RR overshoots seen (printed up to limit):",
          min(dbg_temp, DEBUG_PRINT_LIMIT), "temp;",
          min(dbg_sal, DEBUG_PRINT_LIMIT), "sal")
    return accT, accS

def dump_csv(accT, accS, out_csv):
    rows = []
    keys = sorted(set(list(accT.keys()) + list(accS.keys())))
    for basin, season in keys:
        if (basin, season) in accT:
            tmean, tstd, tn = accT[(basin, season)].finalize()
        else:
            tmean = np.full(len(STD_DEPTHS), np.nan)
            tstd  = np.full(len(STD_DEPTHS), np.nan)
            tn    = np.zeros(len(STD_DEPTHS), dtype=int)

        if (basin, season) in accS:
            smean, sstd, sn = accS[(basin, season)].finalize()
        else:
            smean = np.full(len(STD_DEPTHS), np.nan)
            sstd  = np.full(len(STD_DEPTHS), np.nan)
            sn    = np.zeros(len(STD_DEPTHS), dtype=int)

        for k, z in enumerate(STD_DEPTHS):
            rows.append({
                "basin": basin,
                "season": season,
                "z_m": float(z),
                "n_T": int(tn[k]),
                "T_mean": float(tmean[k]) if np.isfinite(tmean[k]) else np.nan,
                "T_std": float(tstd[k]) if np.isfinite(tstd[k]) else np.nan,
                "n_S": int(sn[k]),
                "S_mean": float(smean[k]) if np.isfinite(smean[k]) else np.nan,
                "S_std": float(sstd[k]) if np.isfinite(sstd[k]) else np.nan,
            })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

def main():
    INFILE = Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins.csv")
    OUTDIR = INFILE.parent

    print("\nSeaDataNet seasonal climatology (multi-period; RR + 1-sigma)")
    print("Input:", str(INFILE))
    print("QC: ON good flags:", sorted(GOOD_QF_NUM))
    print("Cast definition: LOCAL_CDI_ID (fallback Cruise|Station|date if missing)")
    print("Chunk size:", CHUNKSIZE)
    print("Periods:", ", ".join([f"{k} ({v[0]}–{v[1]})" for k, v in PERIODS.items()]))

    for label, (ymin, ymax) in PERIODS.items():
        global YEAR_MIN, YEAR_MAX
        YEAR_MIN, YEAR_MAX = int(ymin), int(ymax)

        out1 = OUTDIR / f"climatology_sdn_{label}_pass1.csv"
        out2 = OUTDIR / f"climatology_sdn_{label}_after_1sigma.csv"

        print("\n" + "-" * 78)
        print(f"Period: {label}  (years {YEAR_MIN}–{YEAR_MAX})")

        print("PASS 1: compute means/stds at standard depths ...")
        accT1, accS1 = pass_stats(str(INFILE), qc_on=True, enforce_depth_qc=False)

        means = {}
        stds  = {}
        for (basin, season), w in accT1.items():
            mu, sd, _ = w.finalize()
            means[("T", basin, season)] = mu
            stds[("T", basin, season)]  = sd
        for (basin, season), w in accS1.items():
            mu, sd, _ = w.finalize()
            means[("S", basin, season)] = mu
            stds[("S", basin, season)]  = sd

        dump_csv(accT1, accS1, str(out1))

        print("PASS 2: reject |x-mu|>1σ and recompute ...")
        accT2, accS2 = pass_stats(
            str(INFILE),
            qc_on=True,
            first_pass_means=means,
            first_pass_stds=stds,
            enforce_depth_qc=False
        )
        dump_csv(accT2, accS2, str(out2))

if __name__ == "__main__":
    main()
