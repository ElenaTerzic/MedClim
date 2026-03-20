#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSL_profiles_dualmode_raw_or_binned.py

A/B test script: choose whether SSL detection uses RAW or BINNED data.

- DETECT_MODE="raw": detect SSL on raw (z,T,S,D) using np.gradient + original-like conditions.
- DETECT_MODE="binned": detect SSL on binned (STD_CENTERS) using bin means + diff gradients + analogous conditions.

The script computes BOTH modes for every cast and writes them to cast_summary.csv, then selects one
based on DETECT_MODE for plotting and summary stats.

This is meant for DJ* basin validation before doing seasonal/platform aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import gsw
    HAS_GSW = True
except Exception:
    HAS_GSW = False

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


# =============================================================================
# SETTINGS
# =============================================================================
TARGET_BASIN = "DJ3"
OUTDIR = Path(f"/Users/elena/Projects/IRB/MedClim/Plots/Adriatic/SSL/OUT_SSL_{TARGET_BASIN}_DUALMODE")

DETECT_MODE    = "raw"   # "raw" or "binned"
PLOT_GRAD_MODE = "raw"   # "raw" or "binned" (set equal to DETECT_MODE during QC)

SEASONS = {"JFM": {1, 2, 3}, "AMJ": {4, 5, 6}, "JAS": {7, 8, 9}, "OND": {10, 11, 12}}

STD_CENTERS = np.array([
    5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200,
    250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,
    1200, 1300, 1400, 1500, 1750, 2000, 2500, 3000,
    3500, 4000, 4500, 5000
], dtype=float)

ZMIN, ZMAX = 0.0, 250.0

SSL_ZMAX = 100.0  # set 100 for Adriatic-shallow metric

CORE_PICK_MODE = "near_target"   # "deepest" or "near_target"
CORE_TARGET_Z  = 100.0

# NOTE: make sure these match your intended SSL metric
SG_THRES   = -0.01   # PSU m-1
SG_MAX_VAL = 0.02     # PSU m-1
SAL_THRES  = 0.05     # PSU
DZ_OFFSET  = 5.0      # m
USE_SSI_CONDITION = True
MIN_RAW_LEVELS = 8

PLOT_SSL_ONLY = True
MAX_PLOTS_PER_PERIOD = 250

MAP_EXTENT = (13.8, 20.6, 39.0, 44.0)

GOOD_QF_SDN  = {"1", "2"}
GOOD_QF_ARGO = {"1", "2"}


# =============================================================================
# INPUT PATHS (edit)
# =============================================================================
SDN_PATH  = Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins.csv")
ARGO_PATH = Path("/Users/elena/Projects/IRB/MedClim/Data/Argo/CSV/Argo_with_basins.csv")

CTD_HR_PATH = Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/CTD_HR_2019_2023_MFSD_long_fixed.csv")
IOR_PATH    = Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/IOR_Adriatic_database_1987_2019_long_fixed.csv")
RBNA_PATH   = Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/RB_Northern_Adriatic_Transect_1979_2018_long_fixed.csv")


# =============================================================================
# BIN HELPERS
# =============================================================================
@dataclass(frozen=True)
class BinScheme:
    centers: np.ndarray
    edges: np.ndarray
    nbins: int


def make_centered_bin_edges(centers: np.ndarray) -> np.ndarray:
    c = np.asarray(centers, float)
    mid = 0.5 * (c[1:] + c[:-1])
    edges = np.empty(c.size + 1, float)
    edges[1:-1] = mid
    edges[0] = c[0] - (mid[0] - c[0])
    edges[-1] = c[-1] + (c[-1] - mid[-1])
    return edges


SCHEME = BinScheme(
    centers=STD_CENTERS,
    edges=make_centered_bin_edges(STD_CENTERS),
    nbins=len(STD_CENTERS),
)

Z_MID = 0.5 * (SCHEME.centers[1:] + SCHEME.centers[:-1])
DZ = (SCHEME.centers[1:] - SCHEME.centers[:-1])


# =============================================================================
# UTILS
# =============================================================================
def month_to_season(m: int) -> Optional[str]:
    for s, months in SEASONS.items():
        if m in months:
            return s
    return None


def qc_mask(df: pd.DataFrame, qc_col: Optional[str], good: Optional[Set[str]]) -> np.ndarray:
    """
    Returns a boolean mask length len(df) indicating whether QC is acceptable.
    Works for QC stored as str/object, int, float, numpy arrays; uses first char.
    """
    if qc_col is None or good is None or qc_col not in df.columns:
        return np.ones(len(df), dtype=bool)

    q = df[qc_col]

    # Convert to pandas Series of strings safely
    if isinstance(q, np.ndarray):
        q = pd.Series(q)
    else:
        q = q.copy()

    q = q.astype(str).str.strip()
    # Use first character when QC comes like "1", "1 " or "1;..." etc.
    q = q.str[0]
    return q.isin(list(good)).to_numpy()


def robust_first_valid(df: pd.DataFrame, col: Optional[str]) -> float:
    if col is None or col not in df.columns:
        return np.nan
    v = pd.to_numeric(df[col], errors="coerce").to_numpy()
    i = np.where(np.isfinite(v))[0]
    return float(v[i[0]]) if i.size else np.nan


def make_adr_cast_id(g: pd.DataFrame) -> str:
    """
    Stable cast id for Adriatic in-situ sources.
    Group key is TIME + STATION (+ rounded lon/lat to avoid collisions).
    """
    t0 = pd.to_datetime(g["TIME"].iloc[0], errors="coerce")
    st = str(g["STATION"].iloc[0]) if "STATION" in g.columns else "NA"
    lon = float(pd.to_numeric(g.get("LONGITUDE", pd.Series([np.nan])), errors="coerce").iloc[0])
    lat = float(pd.to_numeric(g.get("LATITUDE",  pd.Series([np.nan])), errors="coerce").iloc[0])

    ts = "NaT" if pd.isna(t0) else t0.strftime("%Y%m%dT%H%M%S")
    if np.isfinite(lon) and np.isfinite(lat):
        return f"ADR|{st}|{ts}|{lon:.3f}|{lat:.3f}"
    return f"ADR|{st}|{ts}"


# =============================================================================
# SOURCES
# =============================================================================
@dataclass
class SourceSpec:
    name: str
    path: Path
    time_col: str
    basin_col: str
    depth_col: str
    lon_col: str
    lat_col: str
    temp_col: str
    sal_col: str
    temp_qc_col: Optional[str] = None
    sal_qc_col: Optional[str] = None
    good_qf: Set[str] = None  # set of good QC flags
    cast_id_cols: Optional[List[str]] = None
    cast_id_builder: Optional[Callable[[pd.DataFrame], str]] = None
    dens_col: Optional[str] = None
    add_date_to_cast_id: bool = False


def build_cast_id(g: pd.DataFrame, spec: SourceSpec) -> str:
    """
    Build cast ID using:
    - cast_id_builder if provided,
    - else cast_id_cols,
    - else fall back to source name.
    """
    if spec.cast_id_builder is not None:
        return spec.cast_id_builder(g)

    cols = spec.cast_id_cols or []
    parts: List[str] = []
    for c in cols:
        if c in g.columns:
            v = str(g[c].iloc[0])
            if v and v.lower() not in ("nan", "none"):
                parts.append(v.strip())
    return "|".join(parts) if parts else spec.name


SOURCES: List[SourceSpec] = [
    SourceSpec(
        name="SDN",
        path=SDN_PATH,
        time_col="yyyy-mm-ddThh:mm:ss.sss",
        basin_col="basin",
        depth_col="Depth [m]",
        lon_col="Longitude [degrees_east]",
        lat_col="Latitude [degrees_north]",
        temp_col="ITS-90 water temperature [degrees C]",
        sal_col="Water body salinity [per mille]",
        temp_qc_col="QV:SEADATANET.1",
        sal_qc_col="QV:SEADATANET.2",
        good_qf=GOOD_QF_SDN,
        cast_id_cols=["LOCAL_CDI_ID", "Cruise", "Station:METAVAR:INDEXED_TEXT"],
        add_date_to_cast_id=False,
        dens_col=None,
    ),

    # Adriatic regional sets
    SourceSpec(
        name="ADR_CTD_HR",
        path=CTD_HR_PATH,
        time_col="TIME",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        temp_qc_col=None,
        sal_qc_col=None,
        good_qf=set(),  # unused when qc cols are None
        cast_id_cols=["TIME", "STATION", "LONGITUDE", "LATITUDE"],
        cast_id_builder=make_adr_cast_id,
        dens_col=None,
    ),
    SourceSpec(
        name="ADR_IOR",
        path=IOR_PATH,
        time_col="TIME",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        temp_qc_col=None,
        sal_qc_col=None,
        good_qf=set(),
        cast_id_cols=["TIME", "STATION", "LONGITUDE", "LATITUDE"],
        cast_id_builder=make_adr_cast_id,
        dens_col=None,
    ),
    SourceSpec(
        name="ADR_RB",
        path=RBNA_PATH,
        time_col="TIME",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        temp_qc_col=None,
        sal_qc_col=None,
        good_qf=set(),
        cast_id_cols=["TIME", "STATION", "LONGITUDE", "LATITUDE"],
        cast_id_builder=make_adr_cast_id,
        dens_col=None,
    ),

    # ARGO
    SourceSpec(
        name="ARGO",
        path=ARGO_PATH,
        time_col="TIME",
        basin_col="basin",
        depth_col="PRES",          # dbar ~ m
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        temp_col="TEMP",
        sal_col="PSAL",
        temp_qc_col="TEMP_QC",
        sal_qc_col="PSAL_QC",
        good_qf=GOOD_QF_ARGO,
        cast_id_cols=["PLATFORM_NUMBER", "CYCLE_NUMBER"],
        add_date_to_cast_id=False,
        dens_col="SIG0",           # use if present in file
    ),
]


# =============================================================================
# BINNING + GRADIENTS
# =============================================================================
def profile_bin_means(z: np.ndarray, v: np.ndarray, edges: np.ndarray, nbins: int) -> np.ndarray:
    out = np.full(nbins, np.nan, dtype=float)
    bi = np.searchsorted(edges, z, side="right") - 1
    ok = (bi >= 0) & (bi < nbins) & np.isfinite(v)
    bi = bi[ok]
    vv = v[ok]
    if bi.size == 0:
        return out
    cnt = np.bincount(bi, minlength=nbins).astype(float)
    s = np.bincount(bi, weights=vv, minlength=nbins).astype(float)
    m = cnt > 0
    out[m] = s[m] / cnt[m]
    return out


def raw_gradients(z: np.ndarray, v: np.ndarray) -> np.ndarray:
    if z.size < 3:
        return np.full_like(v, np.nan, dtype=float)
    return np.gradient(v, z)


def compute_ssi(z: np.ndarray, rho: np.ndarray, idx_core: int, dz_offset: float = DZ_OFFSET) -> float:
    """
    Simple SSI-like integral (as you used before).
    """
    if idx_core <= 1 or z.size < 3:
        return 0.0
    if not np.all(np.isfinite(rho[:idx_core + 1])):
        return 0.0
    g = 9.81
    z0 = float(z[0])
    zSG = float(z[idx_core])
    zG = 0.5 * (zSG + z0) + dz_offset
    try:
        rho_zG = float(np.interp(zG, z, rho))
    except Exception:
        return 0.0
    SSI = 0.0
    for iz in range(0, idx_core):
        dz = float(z[iz + 1] - z[iz])
        SSI += (float(z[iz]) - zG) * (float(rho[iz]) - rho_zG) * dz
    return SSI * g


def choose_core_index(z: np.ndarray, idx: np.ndarray) -> int:
    if idx.size == 0:
        return -1
    if CORE_PICK_MODE == "deepest":
        return int(idx[np.nanargmax(z[idx])])
    if CORE_PICK_MODE == "near_target":
        return int(idx[np.nanargmin(np.abs(z[idx] - CORE_TARGET_Z))])
    raise ValueError(f"Unknown CORE_PICK_MODE={CORE_PICK_MODE!r}")


def binned_gradients(Tbin: np.ndarray, Sbin: np.ndarray, Dbin: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    TG = np.full(SCHEME.nbins - 1, np.nan)
    SG = np.full(SCHEME.nbins - 1, np.nan)
    DG = np.full(SCHEME.nbins - 1, np.nan)

    okS = np.isfinite(Sbin[:-1]) & np.isfinite(Sbin[1:])
    okT = np.isfinite(Tbin[:-1]) & np.isfinite(Tbin[1:])
    okD = np.isfinite(Dbin[:-1]) & np.isfinite(Dbin[1:])

    SG[okS] = (Sbin[1:][okS] - Sbin[:-1][okS]) / DZ[okS]
    TG[okT] = (Tbin[1:][okT] - Tbin[:-1][okT]) / DZ[okT]
    DG[okD] = (Dbin[1:][okD] - Dbin[:-1][okD]) / DZ[okD]
    return TG, SG, DG


# =============================================================================
# CAST ARRAYS
# =============================================================================
def binned_and_raw_cast_arrays(
    g: pd.DataFrame, spec: SourceSpec
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = pd.to_numeric(g[spec.depth_col], errors="coerce").to_numpy(dtype=float)
    mZ = np.isfinite(z) & (z >= ZMIN) & (z <= ZMAX)
    if not np.any(mZ):
        nanb = np.full(SCHEME.nbins, np.nan)
        return nanb, nanb, nanb, False, np.array([]), np.array([]), np.array([]), np.array([])

    T = pd.to_numeric(g[spec.temp_col], errors="coerce").to_numpy(dtype=float)
    S = pd.to_numeric(g[spec.sal_col], errors="coerce").to_numpy(dtype=float)

    mT = mZ & np.isfinite(T) & qc_mask(g, spec.temp_qc_col, spec.good_qf if spec.good_qf else None)
    mS = mZ & np.isfinite(S) & qc_mask(g, spec.sal_qc_col, spec.good_qf if spec.good_qf else None)

    # raw arrays (aligned on mZ)
    z_raw = z[mZ].astype(float)
    T_raw = np.where(mZ, T, np.nan)[mZ].astype(float)
    S_raw = np.where(mZ, S, np.nan)[mZ].astype(float)

    # raw density
    D_raw = np.full(z_raw.size, np.nan, dtype=float)

    # if density column exists, use it
    if spec.dens_col is not None and spec.dens_col in g.columns:
        D_all = pd.to_numeric(g[spec.dens_col], errors="coerce").to_numpy(dtype=float)
        D_raw = np.where(mZ, D_all, np.nan)[mZ].astype(float)

    # else compute sigma0 from T/S (if possible)
    elif HAS_GSW:
        mTS = mT & mS
        if np.any(mTS):
            lon0 = robust_first_valid(g, spec.lon_col)
            lat0 = robust_first_valid(g, spec.lat_col)
            lon0 = 0.0 if not np.isfinite(lon0) else lon0
            lat0 = 0.0 if not np.isfinite(lat0) else lat0
            try:
                p = z[mTS]
                SA = gsw.SA_from_SP(S[mTS], p, np.full(p.size, lon0), np.full(p.size, lat0))
                CT = gsw.CT_from_t(SA, T[mTS], p)
                sig0 = gsw.sigma0(SA, CT)

                idx_all = np.where(mZ)[0]
                idx_ts = np.where(mTS)[0]
                pos = {int(ii): kk for kk, ii in enumerate(idx_all)}
                for ii, val in zip(idx_ts, sig0):
                    k = pos.get(int(ii), None)
                    if k is not None:
                        D_raw[k] = float(val)
            except Exception:
                pass

    # binned arrays
    Tbin = profile_bin_means(z[mT], T[mT], SCHEME.edges, SCHEME.nbins) if np.any(mT) else np.full(SCHEME.nbins, np.nan)
    Sbin = profile_bin_means(z[mS], S[mS], SCHEME.edges, SCHEME.nbins) if np.any(mS) else np.full(SCHEME.nbins, np.nan)

    Dbin = np.full(SCHEME.nbins, np.nan)
    if spec.dens_col is not None and spec.dens_col in g.columns:
        D_all = pd.to_numeric(g[spec.dens_col], errors="coerce").to_numpy(dtype=float)
        mD = mZ & np.isfinite(D_all)
        if np.any(mD):
            Dbin = profile_bin_means(z[mD], D_all[mD], SCHEME.edges, SCHEME.nbins)
    elif HAS_GSW:
        # optional: bin density from computed D_raw
        mDr = np.isfinite(D_raw) & (z_raw >= ZMIN) & (z_raw <= ZMAX)
        if np.any(mDr):
            Dbin = profile_bin_means(z_raw[mDr], D_raw[mDr], SCHEME.edges, SCHEME.nbins)

    return Tbin, Sbin, Dbin, True, z_raw, T_raw, S_raw, D_raw


# =============================================================================
# SSL DETECTION (RAW vs BINNED)
# =============================================================================
def ssl_core_from_raw(z_raw: np.ndarray, S_raw: np.ndarray, T_raw: np.ndarray, D_raw: np.ndarray) -> Tuple[int, float, float, float, float, float]:
    m = np.isfinite(z_raw) & np.isfinite(S_raw) & (z_raw >= ZMIN) & (z_raw <= SSL_ZMAX)
    if m.sum() < MIN_RAW_LEVELS:
        return -1, np.nan, np.nan, np.nan, np.nan, 0.0

    z = z_raw[m].astype(float)
    S = S_raw[m].astype(float)
    T = T_raw[m].astype(float) if np.any(np.isfinite(T_raw[m])) else np.full_like(S, np.nan)
    D = D_raw[m].astype(float) if np.any(np.isfinite(D_raw[m])) else np.full_like(S, np.nan)

    order = np.argsort(z)
    z, S, T, D = z[order], S[order], T[order], D[order]

    _, uniq = np.unique(z, return_index=True)
    uniq = np.sort(uniq)
    z, S, T, D = z[uniq], S[uniq], T[uniq], D[uniq]

    if z.size < MIN_RAW_LEVELS:
        return -1, np.nan, np.nan, np.nan, np.nan, 0.0

    SG = raw_gradients(z, S)
    TG = raw_gradients(z, T) if np.any(np.isfinite(T)) else np.full_like(SG, np.nan)
    DG = raw_gradients(z, D) if np.any(np.isfinite(D)) else np.full_like(SG, np.nan)

    idx = np.where(np.isfinite(SG) & (SG < SG_THRES))[0]
    if idx.size == 0:
        return -1, np.nan, np.nan, np.nan, np.nan, 0.0

    idx_core = choose_core_index(z, idx)
    z_core = float(z[idx_core])

    cond_depth = z_core <= SSL_ZMAX
    cond_SG_above = np.nanmax(SG[:idx_core]) < SG_MAX_VAL if idx_core >= 2 else False
    cond_SalDiff = (float(S[idx_core]) + SAL_THRES) < float(S[0])

    SSI = 0.0
    cond_SSI = True
    if USE_SSI_CONDITION and np.any(np.isfinite(D)):
        SSI = compute_ssi(z, D, idx_core, dz_offset=DZ_OFFSET)
        cond_SSI = SSI > 0.0

    if not (cond_depth and cond_SG_above and cond_SalDiff and cond_SSI):
        return -1, np.nan, np.nan, np.nan, np.nan, SSI

    return (
        idx_core,
        z_core,
        float(SG[idx_core]),
        float(TG[idx_core]) if np.isfinite(TG[idx_core]) else np.nan,
        float(DG[idx_core]) if np.isfinite(DG[idx_core]) else np.nan,
        SSI,
    )


def ssl_core_from_binned(z_centers: np.ndarray, Sbin: np.ndarray, Tbin: np.ndarray, Dbin: np.ndarray) -> Tuple[int, float, float, float, float, float]:
    TG, SG, DG = binned_gradients(Tbin, Sbin, Dbin)

    m2 = np.isfinite(SG) & (Z_MID <= SSL_ZMAX) & (SG < SG_THRES)
    idx = np.where(m2)[0]
    if idx.size == 0:
        return -1, np.nan, np.nan, np.nan, np.nan, 0.0

    idx_core = choose_core_index(Z_MID, idx)
    z_core = float(Z_MID[idx_core])

    cond_depth = z_core <= SSL_ZMAX
    cond_SG_above = np.nanmax(SG[:idx_core]) < SG_MAX_VAL if idx_core >= 2 else False

    surf_idx = np.where(np.isfinite(Sbin) & (z_centers <= SSL_ZMAX))[0]
    if surf_idx.size == 0:
        return -1, np.nan, np.nan, np.nan, np.nan, 0.0
    i0 = int(surf_idx[0])

    # map gradient index to salinity bin index (approx)
    iS_core = int(min(idx_core, SCHEME.nbins - 2))
    cond_SalDiff = (float(Sbin[iS_core]) + SAL_THRES) < float(Sbin[i0])

    SSI = 0.0
    cond_SSI = True
    if USE_SSI_CONDITION and np.any(np.isfinite(Dbin)):
        max_i = min(iS_core + 1, SCHEME.nbins - 1)
        z_use = z_centers[:max_i + 1]
        d_use = Dbin[:max_i + 1]
        if np.isfinite(d_use).sum() >= 3:
            SSI = compute_ssi(z_use, d_use, idx_core, dz_offset=DZ_OFFSET)
            cond_SSI = SSI > 0.0
        else:
            cond_SSI = False

    if not (cond_depth and cond_SG_above and cond_SalDiff and cond_SSI):
        return -1, np.nan, np.nan, np.nan, np.nan, SSI

    return (
        idx_core,
        z_core,
        float(SG[idx_core]),
        float(TG[idx_core]) if np.isfinite(TG[idx_core]) else np.nan,
        float(DG[idx_core]) if np.isfinite(DG[idx_core]) else np.nan,
        SSI,
    )


# =============================================================================
# PLOTTING
# =============================================================================
def plot_cast_figure(
    out_png: Path,
    meta: Dict,
    Tbin: np.ndarray, Sbin: np.ndarray, Dbin: np.ndarray,
    z_raw: np.ndarray, T_raw: np.ndarray, S_raw: np.ndarray, D_raw: np.ndarray,
    TG_b: np.ndarray, SG_b: np.ndarray, DG_b: np.ndarray
) -> None:
    fig = plt.figure(figsize=(16, 5))
    # more headroom for suptitle
    fig.subplots_adjust(top=0.82)
    gs = fig.add_gridspec(1, 4, width_ratios=[3, 3, 3, 2], wspace=0.35)

    core_depth = meta.get("z_core", np.nan)

    _zr = np.asarray(z_raw, dtype=float)
    zmax_plot = float(np.nanmax(_zr)) if np.isfinite(_zr).any() else ZMAX
    zmax_plot = float(min(ZMAX, max(30.0, np.ceil((zmax_plot + 5.0) / 5.0) * 5.0)))

    def plot_grad_raw(ax2, raw_z, raw_v, color):
        m = np.isfinite(raw_z) & np.isfinite(raw_v)
        if m.sum() < 3:
            return False
        zz = raw_z[m].astype(float)
        vv = raw_v[m].astype(float)
        order = np.argsort(zz)
        zz = zz[order]
        vv = vv[order]
        _, u = np.unique(zz, return_index=True)
        u = np.sort(u)
        zz = zz[u]
        vv = vv[u]
        if zz.size < 3:
            return False
        gg = np.gradient(vv, zz)
        ax2.plot(gg, zz, "--", linewidth=1.5, color=color)
        return True

    def panel(ax, var_bin, color, title, xlabel_bot, xlabel_top, raw_z, raw_v, grad_binned, zmid_binned, add_thresh=False):
        ax.plot(var_bin, SCHEME.centers, color=color, linewidth=2.0)

        # markers at raw points
        m = np.isfinite(raw_z) & np.isfinite(raw_v)
        if np.any(m):
            ax.plot(raw_v[m], raw_z[m], linestyle="None", marker="o", markersize=3.2, color=color, alpha=0.6)

        ax.invert_yaxis()
        ax.set_ylim(zmax_plot, 0.0)
        ax.set_ylabel("Depth (m)")
        ax.set_xlabel(xlabel_bot)
        ax.set_title(title)

        ax2 = ax.twiny()
        ax2.set_xlabel(xlabel_top)

        if PLOT_GRAD_MODE == "raw":
            ok = plot_grad_raw(ax2, raw_z, raw_v, color=color)
            if (not ok) and (grad_binned is not None):
                ax2.plot(grad_binned, zmid_binned, "--", linewidth=1.5, color=color)
        else:
            if grad_binned is not None:
                ax2.plot(grad_binned, zmid_binned, "--", linewidth=1.5, color=color)

        if add_thresh:
            ax2.axvline(SG_THRES, linestyle="-.", linewidth=1.2, color="0.3")

        if np.isfinite(core_depth):
            ax.axhline(core_depth, linestyle=":", linewidth=1.0, color="0.3")

        return ax2

    axT = fig.add_subplot(gs[0, 0])
    panel(axT, Tbin, "red",
          f"T | {meta.get('period','')} {meta.get('season','')}",
          "T (°C)", "dT/dz (°C m$^{-1}$)",
          z_raw, T_raw, TG_b, Z_MID, add_thresh=False)

    axS = fig.add_subplot(gs[0, 1])
    panel(axS, Sbin, "blue",
          f"S | SG_thres={SG_THRES:g} ({meta.get('detect_mode','')})",
          "S (PSU)", "dS/dz (PSU m$^{-1}$)",
          z_raw, S_raw, SG_b, Z_MID, add_thresh=True)

    axD = fig.add_subplot(gs[0, 2])
    panel(axD, Dbin, "black",
          "PDA",
          "PDA (σ0, kg m$^{-3}$)", "dPDA/dz (kg m$^{-4}$)",
          z_raw, D_raw, DG_b, Z_MID, add_thresh=False)

    lon, lat = meta.get("lon", np.nan), meta.get("lat", np.nan)
    if HAS_CARTOPY:
        axM = fig.add_subplot(gs[0, 3], projection=ccrs.PlateCarree())
        axM.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        axM.add_feature(cfeature.LAND, zorder=0)
        axM.add_feature(cfeature.COASTLINE, linewidth=0.8)
        axM.add_feature(cfeature.BORDERS, linewidth=0.4)
        axM.set_title("Location")
        if np.isfinite(lon) and np.isfinite(lat):
            axM.scatter([lon], [lat], transform=ccrs.PlateCarree(), s=35)
    else:
        axM = fig.add_subplot(gs[0, 3])
        axM.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
        axM.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
        axM.set_xlabel("lon"); axM.set_ylabel("lat")
        axM.set_title("Location")
        if np.isfinite(lon) and np.isfinite(lat):
            axM.scatter([lon], [lat], s=35)
        axM.set_aspect("equal", adjustable="box")

    # Clean suptitle
    cast_id = str(meta.get("cast_id", ""))
    cast_short = cast_id[:60] + ("…" if len(cast_id) > 60 else "")
    fig.suptitle(
        f"{TARGET_BASIN} | {meta.get('date_str','')} | {cast_short} | SSL={meta.get('has_ssl',0)}",
        y=0.98, fontsize=11
    )

    # Tight layout leaving room for suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.80])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "profiles" / "20c").mkdir(parents=True, exist_ok=True)
    (OUTDIR / "profiles" / "21c").mkdir(parents=True, exist_ok=True)

    cast_rows: List[Dict] = []

    for spec in SOURCES:
        if not spec.path.exists():
            print(f"[WARN] missing {spec.path} (skip)")
            continue

        print(f"[{spec.name}] reading {spec.path}")
        df = pd.read_csv(spec.path, low_memory=False)

        if spec.basin_col not in df.columns:
            print(f"  [WARN] missing basin_col={spec.basin_col!r} in file; skip source.")
            continue

        # basin filter
        basin = df[spec.basin_col].astype(str).str.strip()
        df = df[basin == TARGET_BASIN].copy()
        if df.empty:
            print("  -> no rows in this basin")
            continue

        # time parsing
        if spec.time_col not in df.columns:
            print(f"  [WARN] missing time_col={spec.time_col!r} in file; skip source.")
            continue

        df["_time"] = pd.to_datetime(df[spec.time_col], errors="coerce")
        df = df[pd.notna(df["_time"])].copy()
        df["_year"] = df["_time"].dt.year.astype(int)
        df["_month"] = df["_time"].dt.month.astype(int)
        df["_season"] = df["_month"].map(lambda m: month_to_season(int(m)))
        df = df[df["_season"].notna()].copy()
        df["_period"] = np.where(df["_year"] < 2000, "20c", "21c")

        # composite grouping key:
        # Prefer cast_id_cols if provided, else fall back to (time + lon/lat)
        cols = spec.cast_id_cols or []
        use_cols = [c for c in cols if c in df.columns]
        if use_cols:
            df["_gkey"] = df[use_cols].astype(str).agg("|".join, axis=1)
        else:
            lon = pd.to_numeric(df.get(spec.lon_col), errors="coerce")
            lat = pd.to_numeric(df.get(spec.lat_col), errors="coerce")
            df["_gkey"] = (
                df["_time"].dt.strftime("%Y-%m-%dT%H:%M")
                + "|" + lon.round(4).astype(str)
                + "|" + lat.round(4).astype(str)
            )

        print(f"  rows after basin/time filter: {len(df):,}")
        print(f"  unique casts (group keys): {df['_gkey'].nunique():,}")

        plotted = {"20c": 0, "21c": 0}

        for _, g in df.groupby("_gkey", sort=False):
            cast_id = build_cast_id(g, spec)
            t0 = pd.to_datetime(g["_time"].iloc[0])
            season = str(g["_season"].iloc[0])
            period = str(g["_period"].iloc[0])
            lon0 = robust_first_valid(g, spec.lon_col)
            lat0 = robust_first_valid(g, spec.lat_col)

            Tbin, Sbin, Dbin, ok, z_raw, T_raw, S_raw, D_raw = binned_and_raw_cast_arrays(g, spec)
            if not ok:
                continue

            TG_b, SG_b, DG_b = binned_gradients(Tbin, Sbin, Dbin)

            # compute both modes
            idxr, zr, SGr, TGr, DGr, SSIr = ssl_core_from_raw(z_raw, S_raw, T_raw, D_raw)
            idxb, zb, SGb, TGb, DGb, SSIb = ssl_core_from_binned(SCHEME.centers, Sbin, Tbin, Dbin)

            if DETECT_MODE == "raw":
                idx_core, z_core, SG_core, TG_core, DG_core, SSI = idxr, zr, SGr, TGr, DGr, SSIr
            elif DETECT_MODE == "binned":
                idx_core, z_core, SG_core, TG_core, DG_core, SSI = idxb, zb, SGb, TGb, DGb, SSIb
            else:
                raise ValueError("DETECT_MODE must be 'raw' or 'binned'")

            has_ssl = int(idx_core >= 0)

            cast_rows.append(dict(
                source=spec.name,
                cast_id=cast_id,
                time=t0,
                year=int(t0.year) if pd.notna(t0) else np.nan,
                season=season,
                period=period,
                lon=lon0,
                lat=lat0,

                # selected mode
                detect_mode=DETECT_MODE,
                has_ssl=has_ssl,
                z_core=z_core,
                SG_core=SG_core,
                TG_core=TG_core,
                DG_core=DG_core,
                SSI=SSI,

                # raw
                has_ssl_raw=int(idxr >= 0),
                z_core_raw=zr,
                SG_core_raw=SGr,
                TG_core_raw=TGr,
                DG_core_raw=DGr,
                SSI_raw=SSIr,

                # binned
                has_ssl_binned=int(idxb >= 0),
                z_core_binned=zb,
                SG_core_binned=SGb,
                TG_core_binned=TGb,
                DG_core_binned=DGb,
                SSI_binned=SSIb,
            ))

            if PLOT_SSL_ONLY and not has_ssl:
                continue
            if plotted[period] >= MAX_PLOTS_PER_PERIOD:
                continue

            meta = dict(
                cast_id=cast_id,
                date_str=t0.strftime("%Y-%m-%d %H:%M"),
                season=season,
                period=period,
                lon=lon0,
                lat=lat0,
                has_ssl=has_ssl,
                z_core=z_core,
                detect_mode=DETECT_MODE,
            )

            safe_id = cast_id.replace("/", "_").replace(" ", "_")
            out_png = OUTDIR / "profiles" / period / f"{t0.strftime('%Y%m%dT%H%M')}_{safe_id[:120]}.png"
            plot_cast_figure(out_png, meta, Tbin, Sbin, Dbin, z_raw, T_raw, S_raw, D_raw, TG_b, SG_b, DG_b)
            plotted[period] += 1

        print(f"  -> plotted: {plotted}")

    cast_df = pd.DataFrame(cast_rows)
    if cast_df.empty:
        print("[ERROR] No casts collected after filtering.")
        return

    cast_df.to_csv(OUTDIR / "cast_summary.csv", index=False)
    print("Wrote:", OUTDIR / "cast_summary.csv")

    # --- Quick totals by source/period to verify in-situ are included
    print("\nTOTAL casts by source/period:")
    print(cast_df.groupby(["source", "period"]).size())

    print("\nSSL counts by source/period:")
    print(cast_df[cast_df.has_ssl == 1].groupby(["source", "period"]).size())

    # --- Seasonal sample size and SSL %
    seasons_order = ["JFM", "AMJ", "JAS", "OND"]
    x = np.arange(len(seasons_order))
    width = 0.35

    grp = cast_df.groupby(["period", "season"]).size().reset_index(name="n_casts")
    c20 = [int(grp[(grp.period == "20c") & (grp.season == s)]["n_casts"].values[0]) if not grp[(grp.period == "20c") & (grp.season == s)].empty else 0 for s in seasons_order]
    c21 = [int(grp[(grp.period == "21c") & (grp.season == s)]["n_casts"].values[0]) if not grp[(grp.period == "21c") & (grp.season == s)].empty else 0 for s in seasons_order]

    plt.figure()
    plt.bar(x - width/2, c20, width, label="20c", color='mediumorchid')
    plt.bar(x + width/2, c21, width, label="21c", color='darkorange')
    plt.xticks(x, seasons_order)
    plt.ylabel("Number of casts")
    #plt.title(f"{TARGET_BASIN}: sample size per season (all casts)")
    plt.title(f"Sample size per season (all casts)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "summary_counts.png", dpi=150)
    plt.close()

    pct20, pct21 = [], []
    for s in seasons_order:
        sub20 = cast_df[(cast_df.period == "20c") & (cast_df.season == s)]
        sub21 = cast_df[(cast_df.period == "21c") & (cast_df.season == s)]
        pct20.append(100.0 * sub20["has_ssl"].mean() if len(sub20) else np.nan)
        pct21.append(100.0 * sub21["has_ssl"].mean() if len(sub21) else np.nan)

    plt.figure()
    plt.bar(x - width/2, pct20, width, label="20c", color='mediumorchid')
    plt.bar(x + width/2, pct21, width, label="21c", color='darkorange')
    plt.xticks(x, seasons_order)
    plt.ylabel("SSL percentage (%)")
    #plt.title(f"{TARGET_BASIN}: SSL occurrence ({DETECT_MODE}, SG<{SG_THRES:g}, z<={SSL_ZMAX:g}m)")
    plt.title(f"SSL occurrence (SG<{SG_THRES:g}, z<={SSL_ZMAX:g}m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "summary_pct_ssl.png", dpi=150)
    plt.close()

    # --- Boxplots for SSL-only casts
    ssl = cast_df[cast_df.has_ssl == 1].copy()

    def seasonal_boxplot(var: str, ylabel: str, outname: str):
        data = []
        labels = []
        for season in seasons_order:
            for period in ["20c", "21c"]:
                vals = ssl[(ssl.season == season) & (ssl.period == period)][var].dropna().values
                data.append(vals)
                labels.append(f"{season}\n{period}")
        plt.figure(figsize=(10, 4))
        plt.boxplot(data)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.ylabel(ylabel)
        plt.title(f"{TARGET_BASIN}: {var} (SSL casts only; {DETECT_MODE})")
        plt.tight_layout()
        plt.savefig(OUTDIR / outname, dpi=150)
        plt.close()

    if not ssl.empty:
        seasonal_boxplot("z_core", "zSG (m)", "box_zSG.png")
        seasonal_boxplot("SG_core", "SG_core (PSU m$^{-1}$)", "box_SG.png")
        seasonal_boxplot("TG_core", "TG_core (°C m$^{-1}$)", "box_TG.png")
        seasonal_boxplot("DG_core", "DG_core (kg m$^{-4}$)", "box_DG.png")

    print("\nDone. Outputs in:", OUTDIR.resolve())


if __name__ == "__main__":
    main()