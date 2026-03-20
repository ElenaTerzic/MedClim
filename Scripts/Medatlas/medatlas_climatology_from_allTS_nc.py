#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import xarray as xr

import Basins  # your polygons module

# ============================================================
# USER SETTINGS
# ============================================================
NC_PATH = "/Users/elena/Projects/IRB/Manca_Argo/Data/Medatlas/NetCDF/allTS_data.nc"
OUT_DIR = "/Users/elena/Projects/IRB/Manca_Argo/Data/Medatlas/Climatology_from_allTS_RR68"
os.makedirs(OUT_DIR, exist_ok=True)

MED_MASK_BASIN = Basins.worldseas.med

YEAR_MIN = 1950
YEAR_MAX = 1999

# Seasons + UNKNOWN bucket for untrusted month metadata
SEASONS = ["JFM", "AMJ", "JAS", "OND", "UNKNOWN"]

# Dedup tolerance
ROUND_LONLAT = 2
TIME_BIN = "D"

# Station filters
MIN_T_SAMPLES = 2
MIN_S_SAMPLES = 1

# Vertical interpolation behavior
FILL_SINGLE_SAMPLE_TO_NEAREST_LEVEL = True
SINGLE_SAMPLE_MAX_DIST_DBAR = 10.0

# Sigma clip
DO_SIGMA_CLIP_1PASS = True
SIGMA_K = 1.0

MIN_N_PER_LEVEL = 5

# ---- TIME/SEASON DIAGNOSTICS / POLICY ----
# If True: compare manual decoding with xarray.decode_cf decoding
DO_TIME_DECODING_CROSSCHECK = True

# If True: months flagged untrusted go to season="UNKNOWN"
USE_UNKNOWN_FOR_UNTRUSTED_MONTHS = True

# If you want to be extremely conservative, you can force month trust to require
# that Jan fraction in 1950-1999 is not crazy. Your file is crazy (~0.6 mean).
# This threshold is "per-year January fraction" above which month assignment is treated as untrusted.
JAN_FRAC_PER_YEAR_MAX_OK = 0.50  # set 0.60 if you want to be looser

# Also require that timestamp fractional-day is not too quantized
MAX_UNIQUE_FRACDAY_OK = 200  # your file has 24 -> will be flagged as low-information

# ============================================================
# DEPTH GRID
# ============================================================
def rr68_levels_coarse():
    return np.array([
        0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100,
        125, 150, 200, 250, 300, 400, 500, 600, 800, 1000,
        1200, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000
    ], dtype=float)

USE_DENSER_ZGRID = False

# ============================================================
# BASINS
# ============================================================
BASIN_LIST = [
    Basins.manca.df2, Basins.manca.df3, Basins.manca.df4, Basins.manca.ds2, Basins.manca.df1,
    Basins.manca.ds1, Basins.manca.ds3, Basins.manca.ds4, Basins.manca.dt1, Basins.manca.dt2,
    Basins.manca.dt3, Basins.manca.di1, Basins.manca.di3, Basins.manca.dj1, Basins.manca.dj2,
    Basins.manca.dj3, Basins.manca.dj6, Basins.manca.dj4, Basins.manca.dj7, Basins.manca.dj8,
    Basins.manca.dj5, Basins.manca.dh1, Basins.manca.dh2, Basins.manca.dh3, Basins.manca.dl1,
    Basins.manca.dl2, Basins.manca.dl3, Basins.manca.dl4
]

def lon_360_to_180(lon):
    return ((lon + 180.0) % 360.0) - 180.0

def assign_manca_basin(lon180, lat):
    xyz = np.array([[lon180, lat, 0.0]], dtype=float)
    for b in BASIN_LIST:
        if b.areinside(xyz)[0]:
            return b.abbrev
    return "none"

# ============================================================
# TIME PARSING
# ============================================================
def parse_odv_datetime_manual(ds):
    """
    Manual decode: interprets date_time as 'days since YYYY-MM-DD HH:MM:SS'.
    Returns datetime64[s] array; invalid values -> NaT.
    """
    days = ds["date_time"].values.astype(float)
    fill = ds["date_time"].attrs.get("_FillValue", None)

    bad = ~np.isfinite(days)
    if fill is not None:
        bad |= (days == fill)

    days = days.copy()
    days[bad] = np.nan

    units = ds["date_time"].attrs.get("units", "")
    m = re.search(r"days since (\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", units)
    if not m:
        raise ValueError(f"Cannot parse date_time units: {units}")

    date0, time0 = m.group(1), m.group(2)
    t0 = pd.Timestamp(f"{date0}T{time0}Z")
    dt = (t0 + pd.to_timedelta(days, unit="D")).to_numpy(dtype="datetime64[s]")
    return dt

def parse_odv_datetime_xarray(ds):
    """
    Decode using CF conventions (xarray + cftime if needed).
    We open a copy with decode_times=True to let xarray interpret units/calendar.
    """
    ds2 = xr.open_dataset(NC_PATH, decode_times=True, use_cftime=True)
    t = ds2["date_time"].values
    # Convert to pandas datetime if possible
    try:
        dt = pd.to_datetime(t).to_numpy(dtype="datetime64[s]")
    except Exception:
        # cftime objects -> keep as object; convert via pandas safe path
        dt = np.array([pd.Timestamp(str(x)).to_datetime64() for x in t], dtype="datetime64[s]")
    return dt

def month_to_season(month):
    if month in (1, 2, 3): return "JFM"
    if month in (4, 5, 6): return "AMJ"
    if month in (7, 8, 9): return "JAS"
    if month in (10, 11, 12): return "OND"
    return "UNKNOWN"

def diagnose_month_trust(ds, dt_idx, year_min, year_max):
    """
    Decide whether month metadata is trustworthy for seasonal binning.

    Uses two objective signals:
      1) Per-year January fraction in the target period (1950-1999)
      2) Quantization of fractional day values in the raw date_time float array
    """
    # 1) January fraction by year in period
    m_y = pd.crosstab(dt_idx.year, dt_idx.month)
    frac_jan = (m_y.get(1, 0) / m_y.sum(axis=1)).fillna(0.0)
    frac_jan_period = frac_jan.loc[(frac_jan.index >= year_min) & (frac_jan.index <= year_max)]
    desc = frac_jan_period.describe()

    # 2) Unique fractional-day count
    days = ds["date_time"].values.astype(float)
    fill = ds["date_time"].attrs.get("_FillValue", None)
    m = np.isfinite(days)
    if fill is not None:
        m &= (days != fill)
    frac = days[m] - np.floor(days[m])
    n_unique_frac = len(np.unique(np.round(frac, 6)))

    # Trust logic
    jan_ok = (desc["mean"] <= JAN_FRAC_PER_YEAR_MAX_OK)
    frac_ok = (n_unique_frac >= MAX_UNIQUE_FRACDAY_OK)

    # We require BOTH to look reasonable to trust months for seasons.
    months_trusted = bool(jan_ok and frac_ok)

    report = {
        "jan_frac_desc_1950_1999": desc,
        "n_unique_fracday": n_unique_frac,
        "months_trusted": months_trusted,
        "jan_ok": bool(jan_ok),
        "frac_ok": bool(frac_ok),
    }
    return months_trusted, report

def build_season_array(dt_idx, months_trusted):
    """
    If months_trusted is False, we still return seasons but mark them UNKNOWN
    (or you can exclude UNKNOWN later).
    """
    months = dt_idx.month.to_numpy(dtype=int)
    if months_trusted:
        seasons = np.array([month_to_season(m) for m in months], dtype=object)
    else:
        # Month is not trusted -> UNKNOWN for all
        seasons = np.full(len(dt_idx), "UNKNOWN", dtype=object)
    return seasons

# ============================================================
# VERTICAL INTERP
# ============================================================
def rr68_parabolic_to_z(p, v, z):
    out = np.full_like(z, np.nan, dtype=float)
    if len(p) == 0:
        return out

    idx = np.argsort(p)
    p = p[idx].astype(float)
    v = v[idx].astype(float)

    if len(p) == 1:
        if FILL_SINGLE_SAMPLE_TO_NEAREST_LEVEL:
            k = int(np.argmin(np.abs(z - p[0])))
            if np.abs(z[k] - p[0]) <= SINGLE_SAMPLE_MAX_DIST_DBAR:
                out[k] = v[0]
        return out

    pmin, pmax = p[0], p[-1]
    inside = (z >= pmin) & (z <= pmax)
    if not np.any(inside):
        return out

    for k in np.where(inside)[0]:
        zk = z[k]
        j = np.searchsorted(p, zk)
        if j == 0 or j >= len(p):
            continue

        i0, i1 = j - 1, j
        cand = [i0, i1]
        if i0 - 1 >= 0: cand.append(i0 - 1)
        if i1 + 1 < len(p): cand.append(i1 + 1)

        cand = np.array(sorted(set(cand)), dtype=int)
        pp = p[cand]
        vv = v[cand]

        if len(pp) < 3:
            out[k] = np.interp(zk, p[[i0, i1]], v[[i0, i1]])
            continue

        w = 1.0 / np.maximum(np.abs(pp - zk), 1e-6)
        A = np.vstack([pp**2, pp, np.ones_like(pp)]).T
        W = np.diag(w)

        try:
            coef = np.linalg.lstsq(W @ A, W @ vv, rcond=None)[0]
            val = coef[0]*zk**2 + coef[1]*zk + coef[2]
        except np.linalg.LinAlgError:
            val = np.interp(zk, p[[i0, i1]], v[[i0, i1]])

        vmin = float(np.nanmin(vv))
        vmax = float(np.nanmax(vv))
        if np.isfinite(vmin) and np.isfinite(vmax):
            val = min(max(val, vmin), vmax)

        out[k] = val

    return out

# ============================================================
# STATS
# ============================================================
def mean_std_from_acc(sum_, sumsq_, N_):
    mean = np.full_like(sum_, np.nan, dtype=float)
    std  = np.full_like(sum_, np.nan, dtype=float)

    ok = N_ >= MIN_N_PER_LEVEL
    mean[ok] = sum_[ok] / N_[ok]
    var = np.full_like(sum_, np.nan, dtype=float)
    var[ok] = (sumsq_[ok] / N_[ok]) - mean[ok]**2
    var = np.maximum(var, 0.0)
    std[ok] = np.sqrt(var[ok])
    return mean, std

def sigma_clip_mask(values_1d, mean_1d, std_1d, k=1.0):
    m = np.isfinite(values_1d) & np.isfinite(mean_1d) & np.isfinite(std_1d)
    m &= (np.abs(values_1d - mean_1d) <= k * std_1d) | (std_1d == 0)
    return m

def make_acc(nz, nseason):
    return dict(
        T_sum=np.zeros((nseason, nz), float),
        T_sumsq=np.zeros((nseason, nz), float),
        N_T=np.zeros((nseason, nz), int),

        T_sum_TS=np.zeros((nseason, nz), float),
        T_sumsq_TS=np.zeros((nseason, nz), float),
        N_T_TS=np.zeros((nseason, nz), int),

        S_sum=np.zeros((nseason, nz), float),
        S_sumsq=np.zeros((nseason, nz), float),
        N_S=np.zeros((nseason, nz), int),
    )

# ============================================================
# MAIN
# ============================================================
def main():
    z = rr68_levels_coarse()
    nz = len(z)
    nseason = len(SEASONS)
    season_to_i = {s: i for i, s in enumerate(SEASONS)}

    ds = xr.open_dataset(NC_PATH, decode_times=False)

    # coords
    lon = ds["longitude"].values.astype(float)
    lat = ds["latitude"].values.astype(float)
    lon180 = lon_360_to_180(lon)

    # MED mask
    xyz = np.zeros((len(lon), 3), dtype=float)
    xyz[:, 0] = lon180
    xyz[:, 1] = lat
    good_xy = np.isfinite(lon180) & np.isfinite(lat)
    m_med = np.zeros(len(lon), dtype=bool)
    m_med[good_xy] = MED_MASK_BASIN.areinside(xyz[good_xy])

    # ---- Decode time (manual)
    dt_manual = parse_odv_datetime_manual(ds)
    dt_idx = pd.DatetimeIndex(dt_manual).round("min")

    # ---- Optional cross-check vs CF decoding
    if DO_TIME_DECODING_CROSSCHECK:
        dt_xr = parse_odv_datetime_xarray(ds)
        dt_xr_idx = pd.DatetimeIndex(dt_xr).round("min")

        same_month = (dt_idx.month == dt_xr_idx.month)
        same_year = (dt_idx.year == dt_xr_idx.year)
        print("TIME CROSSCHECK:")
        print(f"  Same YEAR fraction:  {same_year.mean():.3f}")
        print(f"  Same MONTH fraction: {same_month.mean():.3f}")
        # If these are very low, your manual decoding is wrong or CF decoding differs due to calendar.
        # If these are ~1, decoding is consistent and the issue is likely the metadata itself.

    # ---- Diagnose whether month is trustworthy
    years = dt_idx.year.to_numpy(dtype=float)
    m_time = np.isfinite(years) & (years >= YEAR_MIN) & (years <= YEAR_MAX)
    base = m_med & m_time

    months_trusted, report = diagnose_month_trust(ds, dt_idx, YEAR_MIN, YEAR_MAX)
    print("MONTH TRUST DIAGNOSTICS (1950-1999):")
    print(report["jan_frac_desc_1950_1999"])
    print("Unique fractional-day values:", report["n_unique_fracday"])
    print("Months trusted?:", report["months_trusted"], "(jan_ok:", report["jan_ok"], ", frac_ok:", report["frac_ok"], ")")

    # ---- Build seasons
    season_all = build_season_array(dt_idx, months_trusted=months_trusted)
    if not USE_UNKNOWN_FOR_UNTRUSTED_MONTHS and (not months_trusted):
        # If you prefer to EXCLUDE rather than bucket them:
        # We'll later mask them out by setting base &= False on those.
        pass

    # ---- Memory-safe DataArrays
    P_da = ds["Pressure"]
    T_da = ds["Temperature"]
    S_da = ds["Salinity"] if "Salinity" in ds.variables else None

    OBS_DIM = T_da.dims[0]
    LEV_DIM = T_da.dims[1]

    fillP = P_da.attrs.get("_FillValue", None)
    fillT = T_da.attrs.get("_FillValue", None)
    fillS = S_da.attrs.get("_FillValue", None) if S_da is not None else None

    # counts (1D only)
    vT_da = xr.apply_ufunc(np.isfinite, T_da)
    if fillT is not None:
        vT_da = vT_da & (T_da != fillT)
    nT = vT_da.sum(dim=LEV_DIM).compute().values.astype(int)

    if S_da is not None:
        vS_da = xr.apply_ufunc(np.isfinite, S_da)
        if fillS is not None:
            vS_da = vS_da & (S_da != fillS)
        nS = vS_da.sum(dim=LEV_DIM).compute().values.astype(int)
    else:
        nS = np.zeros(len(lon), dtype=int)

    # If months are untrusted and you want to EXCLUDE rather than bucket:
    if (not months_trusted) and (not USE_UNKNOWN_FOR_UNTRUSTED_MONTHS):
        base = base & (season_all != "UNKNOWN")

    station_ok_T = base & (nT >= MIN_T_SAMPLES)
    station_ok   = base & (nT >= MIN_T_SAMPLES) & (nS >= MIN_S_SAMPLES)

    # Dedup keys
    lon_r = np.round(lon180, ROUND_LONLAT)
    lat_r = np.round(lat, ROUND_LONLAT)
    day_bin = dt_manual.astype(f"datetime64[{TIME_BIN}]").astype("int64")

    dfk = pd.DataFrame({
        "idx": np.arange(len(lon), dtype=int),
        "lon": lon_r,
        "lat": lat_r,
        "day": day_bin,
    })
    dfk = dfk.loc[station_ok_T].copy()
    dfk["key"] = dfk["lon"].astype(str) + "|" + dfk["lat"].astype(str) + "|" + dfk["day"].astype(str)
    dfk = dfk.drop_duplicates(subset=["key"], keep="first")
    keep_idx = dfk["idx"].to_numpy(dtype=int)

    print(f"Stations after Med+time base mask:          {int(np.sum(base)):,}")
    print(f"Stations after TS filters (T>=2 & S>=1):    {int(np.sum(station_ok)):,}")
    print(f"Stations after T-only filters (T>=2):       {int(np.sum(station_ok_T)):,}")
    print(f"Stations after dedup (lon/lat/day, T-only): {len(keep_idx):,}")

    # Basin + season arrays for kept stations
    basins = np.array([assign_manca_basin(lon180[i], lat[i]) for i in keep_idx], dtype=object)
    seasons = season_all[keep_idx].astype(object)

    print("After dedup, season counts:")
    print(pd.Series(seasons).value_counts())

    basin_names = sorted(set(b for b in basins if b != "none"))
    acc1 = {b: make_acc(nz, nseason) for b in basin_names}

    def valid_profile_mask(arr, fill):
        m = np.isfinite(arr)
        if fill is not None:
            m &= (arr != fill)
        return m

    # PASS 1
    for n, i in enumerate(keep_idx, 1):
        b = basins[n - 1]
        if b == "none":
            continue

        si = season_to_i.get(seasons[n - 1], season_to_i["UNKNOWN"])

        p = P_da.isel({OBS_DIM: i}).values.astype(float)
        t = T_da.isel({OBS_DIM: i}).values.astype(float)

        mp = valid_profile_mask(p, fillP)
        mt = valid_profile_mask(t, fillT)
        ms_T = mp & mt

        Tz = rr68_parabolic_to_z(p[ms_T], t[ms_T], z)

        goodT = np.isfinite(Tz)
        if np.any(goodT):
            acc1[b]["T_sum"][si, goodT] += Tz[goodT]
            acc1[b]["T_sumsq"][si, goodT] += Tz[goodT]**2
            acc1[b]["N_T"][si, goodT] += 1

        if station_ok[i] and (S_da is not None):
            sarr = S_da.isel({OBS_DIM: i}).values.astype(float)
            ms = valid_profile_mask(sarr, fillS)
            ms_S = mp & ms

            Sz = rr68_parabolic_to_z(p[ms_S], sarr[ms_S], z)
            goodS = np.isfinite(Sz)
            if np.any(goodS):
                acc1[b]["S_sum"][si, goodS] += Sz[goodS]
                acc1[b]["S_sumsq"][si, goodS] += Sz[goodS]**2
                acc1[b]["N_S"][si, goodS] += 1

            ms_TTS = ms_S & mt
            Tz_TS = rr68_parabolic_to_z(p[ms_TTS], t[ms_TTS], z)
            goodTTS = np.isfinite(Tz_TS)
            if np.any(goodTTS):
                acc1[b]["T_sum_TS"][si, goodTTS] += Tz_TS[goodTTS]
                acc1[b]["T_sumsq_TS"][si, goodTTS] += Tz_TS[goodTTS]**2
                acc1[b]["N_T_TS"][si, goodTTS] += 1

        if n % 20000 == 0:
            print(f"[PASS1] Processed {n:,} / {len(keep_idx):,}")

    # Stats for sigma clip
    stats1 = {}
    for b in basin_names:
        A = acc1[b]
        stats1[b] = {}
        stats1[b]["T_mean1"], stats1[b]["T_std1"] = mean_std_from_acc(A["T_sum"], A["T_sumsq"], A["N_T"])
        stats1[b]["S_mean1"], stats1[b]["S_std1"] = mean_std_from_acc(A["S_sum"], A["S_sumsq"], A["N_S"])
        stats1[b]["T_meanTS1"], stats1[b]["T_stdTS1"] = mean_std_from_acc(A["T_sum_TS"], A["T_sumsq_TS"], A["N_T_TS"])

    # PASS 2
    if DO_SIGMA_CLIP_1PASS:
        acc2 = {b: make_acc(nz, nseason) for b in basin_names}

        for n, i in enumerate(keep_idx, 1):
            b = basins[n - 1]
            if b == "none":
                continue

            si = season_to_i.get(seasons[n - 1], season_to_i["UNKNOWN"])

            p = P_da.isel({OBS_DIM: i}).values.astype(float)
            t = T_da.isel({OBS_DIM: i}).values.astype(float)

            mp = valid_profile_mask(p, fillP)
            mt = valid_profile_mask(t, fillT)
            ms_T = mp & mt

            Tz = rr68_parabolic_to_z(p[ms_T], t[ms_T], z)
            mkeepT = sigma_clip_mask(Tz, stats1[b]["T_mean1"][si], stats1[b]["T_std1"][si], k=SIGMA_K)
            if np.any(mkeepT):
                acc2[b]["T_sum"][si, mkeepT] += Tz[mkeepT]
                acc2[b]["T_sumsq"][si, mkeepT] += Tz[mkeepT]**2
                acc2[b]["N_T"][si, mkeepT] += 1

            if station_ok[i] and (S_da is not None):
                sarr = S_da.isel({OBS_DIM: i}).values.astype(float)
                ms = valid_profile_mask(sarr, fillS)
                ms_S = mp & ms

                Sz = rr68_parabolic_to_z(p[ms_S], sarr[ms_S], z)
                mkeepS = sigma_clip_mask(Sz, stats1[b]["S_mean1"][si], stats1[b]["S_std1"][si], k=SIGMA_K)
                if np.any(mkeepS):
                    acc2[b]["S_sum"][si, mkeepS] += Sz[mkeepS]
                    acc2[b]["S_sumsq"][si, mkeepS] += Sz[mkeepS]**2
                    acc2[b]["N_S"][si, mkeepS] += 1

                ms_TTS = ms_S & mt
                Tz_TS = rr68_parabolic_to_z(p[ms_TTS], t[ms_TTS], z)
                mkeepTTS = sigma_clip_mask(Tz_TS, stats1[b]["T_meanTS1"][si], stats1[b]["T_stdTS1"][si], k=SIGMA_K)
                if np.any(mkeepTTS):
                    acc2[b]["T_sum_TS"][si, mkeepTTS] += Tz_TS[mkeepTTS]
                    acc2[b]["T_sumsq_TS"][si, mkeepTTS] += Tz_TS[mkeepTTS]**2
                    acc2[b]["N_T_TS"][si, mkeepTTS] += 1

            if n % 20000 == 0:
                print(f"[PASS2] Processed {n:,} / {len(keep_idx):,}")

        acc_final = acc2
    else:
        acc_final = acc1

    # WRITE NPZ
    for b in basin_names:
        A = acc_final[b]
        T_mean, T_std = mean_std_from_acc(A["T_sum"], A["T_sumsq"], A["N_T"])
        S_mean, S_std = mean_std_from_acc(A["S_sum"], A["S_sumsq"], A["N_S"])
        T_mean_TS, T_std_TS = mean_std_from_acc(A["T_sum_TS"], A["T_sumsq_TS"], A["N_T_TS"])

        out = os.path.join(OUT_DIR, f"MedatlasII_climatology_{b}_RR68.npz")
        np.savez(
            out,
            basin=b,
            z=z,
            seasons=np.array(SEASONS, dtype=object),
            T_mean=T_mean, T_std=T_std, N_T=A["N_T"],
            T_mean_TS=T_mean_TS, T_std_TS=T_std_TS, N_T_TS=A["N_T_TS"],
            S_mean=S_mean, S_std=S_std, N_S=A["N_S"],
        )
        print("Wrote:", out)

    print("Done.")

if __name__ == "__main__":
    main()
