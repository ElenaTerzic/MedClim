#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSL_profiles.py

Per-cast SSL diagnostics + seasonal stats for ONE basin (set TARGET_BASIN below).
Designed for DJ* basins (DJ1, DJ3, ...), multi-source (SDN + regional CSVs + Argo).

Outputs (in OUTDIR):
- profiles/20c/*.png , profiles/21c/*.png : per-cast 4-column figures (T | S | PDA | Map)
- cast_summary.csv : per-cast metadata + SSL core metrics (z_core, SG_core, TG_core, DG_core)
- summary_counts.png : number of casts per season per century (all casts)
- summary_pct_ssl.png : % SSL per season per century (all casts)
- box_zSG.png, box_SG.png, box_TG.png, box_DG.png : seasonal boxplots (SSL casts only)

Notes:
- Solid lines = BINNED profile
- Dashed lines (top x-axis) = gradient computed from BINNED profile
- Markers = raw observed levels (original vertical resolution)
- SG threshold is applied to BINNED salinity gradient SG in top SSL_ZMAX meters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional density from TEOS-10
try:
    import gsw
    HAS_GSW = True
except Exception:
    HAS_GSW = False

# Optional coastline map
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


# =============================================================================
# SETTINGS (EDIT HERE ONLY)
# =============================================================================
TARGET_BASIN = "DJ3"          # <-- set basin here (DJ1, DJ3, ...)
OUTDIR = Path(f"/Users/elena/Projects/IRB/MedClim/Plots/Adriatic/SSL/OUT_SSL_PLOTS_{TARGET_BASIN}")

# Seasons
SEASONS = {"JFM": {1, 2, 3}, "AMJ": {4, 5, 6}, "JAS": {7, 8, 9}, "OND": {10, 11, 12}}

# ORIGINAL vertical grid (your "beginning" grid)
STD_CENTERS = np.array([
    5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200,
    250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,
    1200, 1300, 1400, 1500, 1750, 2000, 2500, 3000,
    3500, 4000, 4500, 5000
], dtype=float)

# Plotting / SSL settings
ZMIN, ZMAX = 0.0, 250.0        # plot cap; y-limits auto-zoom to cast depth up to this cap
SSL_ZMAX = 200.0               # search SSL only in top 100 m
SG_THRES = -0.01               # first pass
SG_MAX_VAL = 0.02              # like original metric: max SG above core must be < this
SAL_THRES   = 0.05             # like original: surface S must exceed S(zSG) by at least this
DZ_OFFSET   = 5.0              # like original SSI reference depth offset
USE_SSI_CONDITION = True       # apply SSI>0 if density is available (recommended)
MIN_RAW_LEVELS = 8             # require at least this many raw levels in 0–SSL_ZMAX for SSL test

PLOT_SSL_ONLY = True          # plot only SSL casts (True) or all casts (False)
MAX_PLOTS_PER_PERIOD = 250    # cap per century folder (QC sampling)

# Map extent (adjust if needed)
MAP_EXTENT = (13.8, 20.6, 39.0, 44.0)  # lonmin, lonmax, latmin, latmax

GOOD_QF_SDN = {"1", "2"}
GOOD_QF_ARGO = {"1", "2"}

CORE_PICK_MODE = "deepest"   # options: "deepest" or "near_target"
CORE_TARGET_Z  = 200.0       # only used for "near_target"


# =============================================================================
# INPUT PATHS (EDIT AS NEEDED)
# =============================================================================
# SeaDataNet "profiles-only-with-basins" CSV
SDN_PATH = Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins.csv")

# Regional Adriatic CSVs (optional; keep or comment out)
CTD_HR_PATH = Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/CTD_HR_2019_2023_MFSD_long_fixed.csv")
IOR_PATH    = Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/IOR_Adriatic_database_1987_2019_long_fixed.csv")
RBNA_PATH   = Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/RB_Northern_Adriatic_Transect_1979_2018_long_fixed.csv")

# Argo (your merged CSV with basins)
ARGO_PATH  = Path("/Users/elena/Projects/IRB/MedClim/Data/Argo/CSV/Argo_with_basins.csv")


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
    if c.size < 2:
        raise ValueError("Need at least 2 bin centers")
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
    cast_id_cols: List[str]
    add_date_to_cast_id: bool = True
    lon_col: Optional[str] = None
    lat_col: Optional[str] = None
    dens_col: Optional[str] = None  # if density already available (e.g., Argo SIG0)
    temp_qc_col: Optional[str] = None
    sal_qc_col: Optional[str] = None
    dens_qc_col: Optional[str] = None
    good_qf: Optional[Set[str]] = None


SOURCES: List[SourceSpec] = [
    # SeaDataNet (your SDN export)
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
        cast_id_cols=["LOCAL_CDI_ID", "Cruise", "Station:METAVAR:INDEXED_TEXT"],
        add_date_to_cast_id=False,
        temp_qc_col="QV:SEADATANET.1",
        sal_qc_col="QV:SEADATANET.2",
        good_qf=GOOD_QF_SDN,
    ),

    # Regional CTD sets (edit column names only if your files differ)
    SourceSpec(
        name="CTD_HR_2019_2023_MFSD",
        path=CTD_HR_PATH,
        time_col="TIME",
        basin_col="BASIN_3ADR",
        depth_col="DEPTH_M",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        cast_id_cols=["STATION"],
        add_date_to_cast_id=True,
    ),
    SourceSpec(
        name="IOR_Adriatic_database_1987_2019",
        path=IOR_PATH,
        time_col="TIME",
        basin_col="BASIN_3ADR",
        depth_col="DEPTH_M",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        cast_id_cols=["STATION", "CAST_IDX"],
        add_date_to_cast_id=True,
    ),
    SourceSpec(
        name="RB_Northern_Adriatic_Transect_1979_2018",
        path=RBNA_PATH,
        time_col="TIME",
        basin_col="BASIN_3ADR",
        depth_col="DEPTH_M",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        cast_id_cols=["STATION"],
        add_date_to_cast_id=True,
    ),

    # Argo (based on your header: Argo_short.csv)
    SourceSpec(
        name="ARGO",
        path=ARGO_PATH,
        time_col="TIME",
        basin_col="basin",
        depth_col="PRES",          # dbar ~ m for our purposes
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        temp_col="TEMP",
        sal_col="PSAL",
        dens_col="SIG0",           # already computed
        cast_id_cols=["PLATFORM_NUMBER", "CYCLE_NUMBER"],
        add_date_to_cast_id=False,
        temp_qc_col="TEMP_QC",
        sal_qc_col="PSAL_QC",
        good_qf=GOOD_QF_ARGO,
    ),
]


# =============================================================================
# UTILS
# =============================================================================
def month_to_season(m: int) -> Optional[str]:
    for s, months in SEASONS.items():
        if m in months:
            return s
    return None

def qc_mask(df: pd.DataFrame, qc_col: Optional[str], good: Optional[Set[str]]) -> np.ndarray:
    if qc_col is None or good is None or qc_col not in df.columns:
        return np.ones(len(df), dtype=bool)
    q = df[qc_col].astype(str).str.strip().to_numpy()
    return np.isin(q, list(good))

def robust_first_valid(df: pd.DataFrame, col: Optional[str]) -> float:
    if col is None or col not in df.columns:
        return np.nan
    v = pd.to_numeric(df[col], errors="coerce").to_numpy()
    i = np.where(np.isfinite(v))[0]
    return float(v[i[0]]) if i.size else np.nan

def build_cast_id(g: pd.DataFrame, spec: SourceSpec) -> str:
    parts = []
    for c in spec.cast_id_cols:
        if c in g.columns:
            v = str(g[c].iloc[0])
            if v and v.lower() not in ("nan", "none"):
                parts.append(v.strip())
    if spec.add_date_to_cast_id and spec.time_col in g.columns:
        t = pd.to_datetime(g[spec.time_col].iloc[0], errors="coerce")
        if pd.notna(t):
            parts.append(t.strftime("%Y-%m-%d"))
    if not parts:
        lon = robust_first_valid(g, spec.lon_col)
        lat = robust_first_valid(g, spec.lat_col)
        t = pd.to_datetime(g[spec.time_col].iloc[0], errors="coerce")
        parts = [spec.name, f"{lon:.4f}", f"{lat:.4f}", str(t)]
    return "|".join(parts)

def profile_bin_means(z: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.full(SCHEME.nbins, np.nan, dtype=float)
    if z.size == 0:
        return out
    bi = np.searchsorted(SCHEME.edges, z, side="right") - 1
    ok = (bi >= 0) & (bi < SCHEME.nbins) & np.isfinite(v)
    bi = bi[ok]
    vv = v[ok]
    if bi.size == 0:
        return out
    cnt = np.bincount(bi, minlength=SCHEME.nbins).astype(float)
    s = np.bincount(bi, weights=vv, minlength=SCHEME.nbins).astype(float)
    m = cnt > 0
    out[m] = s[m] / cnt[m]
    return out

def binned_cast_arrays(
    g: pd.DataFrame, spec: SourceSpec
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return binned (T,S,D) and raw (z,T,S,D) arrays."""
    z = pd.to_numeric(g[spec.depth_col], errors="coerce").to_numpy(dtype=float)
    mZ = np.isfinite(z) & (z >= ZMIN) & (z <= ZMAX)
    if not np.any(mZ):
        nanb = np.full(SCHEME.nbins, np.nan)
        return (nanb, nanb, nanb, False,
                np.array([], float), np.array([], float), np.array([], float), np.array([], float))

    T = pd.to_numeric(g[spec.temp_col], errors="coerce").to_numpy(dtype=float)
    S = pd.to_numeric(g[spec.sal_col], errors="coerce").to_numpy(dtype=float)

    mT = mZ & np.isfinite(T) & qc_mask(g, spec.temp_qc_col, spec.good_qf)
    mS = mZ & np.isfinite(S) & qc_mask(g, spec.sal_qc_col, spec.good_qf)

    z_raw = z[mZ].astype(float)
    T_raw = np.where(mZ, T, np.nan)[mZ].astype(float)
    S_raw = np.where(mZ, S, np.nan)[mZ].astype(float)

    # Raw density
    D_raw = np.full(z_raw.size, np.nan, dtype=float)
    if spec.dens_col is not None and spec.dens_col in g.columns:
        D_all = pd.to_numeric(g[spec.dens_col], errors="coerce").to_numpy(dtype=float)
        D_raw = np.where(mZ, D_all, np.nan)[mZ].astype(float)
    elif HAS_GSW:
        mTS = mT & mS
        if np.any(mTS):
            lon = robust_first_valid(g, spec.lon_col)
            lat = robust_first_valid(g, spec.lat_col)
            lon = 0.0 if not np.isfinite(lon) else lon
            lat = 0.0 if not np.isfinite(lat) else lat
            try:
                p = z[mTS]
                SA = gsw.SA_from_SP(S[mTS], p, np.full(p.size, lon), np.full(p.size, lat))
                CT = gsw.CT_from_t(SA, T[mTS], p)
                sig0 = gsw.sigma0(SA, CT)
                # map to raw slice
                idx_all = np.where(mZ)[0]
                idx_ts = np.where(mTS)[0]
                pos = {int(ii): kk for kk, ii in enumerate(idx_all)}
                for ii, val in zip(idx_ts, sig0):
                    k = pos.get(int(ii), None)
                    if k is not None:
                        D_raw[k] = float(val)
            except Exception:
                pass

    # Binned arrays
    Tbin = profile_bin_means(z[mT], T[mT]) if np.any(mT) else np.full(SCHEME.nbins, np.nan)
    Sbin = profile_bin_means(z[mS], S[mS]) if np.any(mS) else np.full(SCHEME.nbins, np.nan)

    Dbin = np.full(SCHEME.nbins, np.nan)
    if spec.dens_col is not None and spec.dens_col in g.columns:
        D_all = pd.to_numeric(g[spec.dens_col], errors="coerce").to_numpy(dtype=float)
        mD = mZ & np.isfinite(D_all)
        if np.any(mD):
            Dbin = profile_bin_means(z[mD], D_all[mD])
    elif HAS_GSW and np.any(mT & mS):
        mTS = mT & mS
        lon = robust_first_valid(g, spec.lon_col)
        lat = robust_first_valid(g, spec.lat_col)
        lon = 0.0 if not np.isfinite(lon) else lon
        lat = 0.0 if not np.isfinite(lat) else lat
        try:
            p = z[mTS]
            SA = gsw.SA_from_SP(S[mTS], p, np.full(p.size, lon), np.full(p.size, lat))
            CT = gsw.CT_from_t(SA, T[mTS], p)
            sig0 = gsw.sigma0(SA, CT)
            Dbin = profile_bin_means(z[mTS], sig0)
        except Exception:
            pass

    return Tbin, Sbin, Dbin, True, z_raw, T_raw, S_raw, D_raw

def gradients_from_bins(Tbin: np.ndarray, Sbin: np.ndarray, Dbin: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def find_ssl_core(SG: np.ndarray) -> Tuple[int, float]:
    m = np.isfinite(SG) & (Z_MID <= SSL_ZMAX) & (SG < SG_THRES)
    idx = np.where(m)[0]
    if idx.size == 0:
        return -1, np.nan
    i0 = idx[np.nanargmin(SG[idx])]
    return int(i0), float(Z_MID[i0])


# =============================================================================
# RAW-SPACE SSL METRIC (closer to compute_gradients.py)
# =============================================================================
def raw_gradients(z: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute dv/dz using np.gradient on raw (z,v). Requires monotonic z."""
    if z.size < 3:
        return np.full_like(v, np.nan, dtype=float)
    return np.gradient(v, z)

def compute_ssi(z: np.ndarray, rho: np.ndarray, idx_core: int, dz_offset: float = DZ_OFFSET) -> float:
    """Schmidt Stability Index-like integral from surface to z_core (as in original script).
    Returns 0 if cannot compute.
    """
    if idx_core <= 1 or z.size < 3:
        return 0.0
    if not np.all(np.isfinite(rho[:idx_core+1])):
        return 0.0
    g = 9.81
    z0 = float(z[0])
    zSG = float(z[idx_core])
    zG = 0.5 * (zSG + z0) + dz_offset

    # rho at zG by interpolation (as in original)
    try:
        rho_zG = float(np.interp(zG, z, rho))
    except Exception:
        return 0.0

    SSI = 0.0
    for iz in range(0, idx_core):
        dz = float(z[iz + 1] - z[iz])
        SSI += (float(z[iz]) - zG) * (float(rho[iz]) - rho_zG) * dz

    return SSI * g

def ssl_core_from_raw(
    z_raw: np.ndarray, S_raw: np.ndarray, T_raw: np.ndarray, D_raw: np.ndarray
) -> Tuple[int, float, float, float, float, float]:
    """Return (idx_core, z_core, SG_core, TG_core, DG_core, SSI) using original-like conditions."""
    # mask to 0–SSL_ZMAX and finite S
    m = np.isfinite(z_raw) & np.isfinite(S_raw) & (z_raw >= ZMIN) & (z_raw <= SSL_ZMAX)
    if m.sum() < MIN_RAW_LEVELS:
        return -1, np.nan, np.nan, np.nan, np.nan, 0.0

    # sort by depth
    z = z_raw[m].astype(float)
    S = S_raw[m].astype(float)
    T = T_raw[m].astype(float) if np.any(np.isfinite(T_raw[m])) else np.full_like(S, np.nan)
    D = D_raw[m].astype(float) if np.any(np.isfinite(D_raw[m])) else np.full_like(S, np.nan)

    order = np.argsort(z)
    z, S, T, D = z[order], S[order], T[order], D[order]

    # require strictly increasing depths (deduplicate if necessary)
    _, uniq = np.unique(z, return_index=True)
    uniq = np.sort(uniq)
    z, S, T, D = z[uniq], S[uniq], T[uniq], D[uniq]

    if z.size < MIN_RAW_LEVELS:
        return -1, np.nan, np.nan, np.nan, np.nan, 0.0

    SG = raw_gradients(z, S)
    TG = raw_gradients(z, T) if np.any(np.isfinite(T)) else np.full_like(SG, np.nan)
    DG = raw_gradients(z, D) if np.any(np.isfinite(D)) else np.full_like(SG, np.nan)

    # candidate indices where SG < threshold
    idx = np.where(np.isfinite(SG) & (SG < SG_THRES))[0]
    if idx.size == 0:
        return -1, np.nan, np.nan, np.nan, np.nan, 0.0

    # choose the SSL core index
    # (match original behavior: deepest threshold-crossing, or closest to a target depth)
    if CORE_PICK_MODE == "deepest":
        idx_core = int(idx[np.nanargmax(z[idx])])
    elif CORE_PICK_MODE == "near_target":
        idx_core = int(idx[np.nanargmin(np.abs(z[idx] - CORE_TARGET_Z))])
    else:
        raise ValueError(f"Unknown CORE_PICK_MODE={CORE_PICK_MODE!r}")

    z_core = float(z[idx_core])

    # Conditions (original-like)
    cond_depth = z_core <= SSL_ZMAX
    #cond_SG_above = np.nanmax(SG[:idx_core+1]) < SG_MAX_VAL if idx_core >= 1 else False
    #ompute cond_SG_above excluding the core point itself, 
    #so the negative core doesn’t trivially satisfy/affect the max:
    cond_SG_above = np.nanmax(SG[:idx_core]) < SG_MAX_VAL if idx_core >= 2 else False
    cond_SalDiff = (float(S[idx_core]) + SAL_THRES) < float(S[0])

    SSI = 0.0
    cond_SSI = True
    if USE_SSI_CONDITION and np.any(np.isfinite(D)):
        SSI = compute_ssi(z, D, idx_core, dz_offset=DZ_OFFSET)
        cond_SSI = SSI > 0.0

    if not (cond_depth and cond_SG_above and cond_SalDiff and cond_SSI):
        return -1, np.nan, np.nan, np.nan, np.nan, SSI

    SG_core = float(SG[idx_core])
    TG_core = float(TG[idx_core]) if np.isfinite(TG[idx_core]) else np.nan
    DG_core = float(DG[idx_core]) if np.isfinite(DG[idx_core]) else np.nan
    return idx_core, z_core, SG_core, TG_core, DG_core, SSI

# =============================================================================
# PLOTTING
# =============================================================================
def plot_cast_figure(
    out_png: Path, meta: Dict,
    Tbin, Sbin, Dbin, TG, SG, DG, i_core: int,
    z_raw: np.ndarray, T_raw: np.ndarray, S_raw: np.ndarray, D_raw: np.ndarray,
):
    """4 columns: [T] [S] [PDA] [Map] with raw markers and binned gradients."""
    fig = plt.figure(figsize=(16, 5))
    # leave room for the suptitle
    fig.subplots_adjust(top=0.84)
    gs = fig.add_gridspec(1, 4, width_ratios=[3, 3, 3, 2], wspace=0.35)

    core_depth = meta.get("z_core", np.nan)

    # dynamic depth zoom (shallow casts won't be squashed)
    zmax_cast = np.nanmax(z_raw) if z_raw.size else ZMAX
    zmax_plot = min(ZMAX, max(10.0, float(zmax_cast) + 2.0))

    def panel(ax, z_centers, var_bin, zmid, grad, xlabel_bot, xlabel_top, title, color, raw_z, raw_v, add_sg_thresh=False):
        ax.plot(var_bin, z_centers, color=color, linewidth=2.0)
        if raw_z.size and raw_v.size:
            m = np.isfinite(raw_z) & np.isfinite(raw_v)
            if np.any(m):
                ax.plot(raw_v[m], raw_z[m], linestyle="None", marker="o",
                        markersize=3.5, color=color, alpha=0.7)

        ax.invert_yaxis()
        ax.set_ylim(zmax_plot, ZMIN)
        ax.set_ylabel("Depth (m)")
        ax.set_xlabel(xlabel_bot)
        ax.set_title(title)

        ax2 = ax.twiny()

        # Plot RAW gradients as dashed (so the figure matches SSL detection done on raw points).
        # Fallback to binned gradients only if raw is insufficient.
        grad_plotted = False
        if raw_z.size and raw_v.size:
            m2 = np.isfinite(raw_z) & np.isfinite(raw_v)
            if np.any(m2) and m2.sum() >= 3:
                zz = raw_z[m2].astype(float)
                vv = raw_v[m2].astype(float)
                order2 = np.argsort(zz)
                zz = zz[order2]
                vv = vv[order2]
                # unique depths
                _, u2 = np.unique(zz, return_index=True)
                u2 = np.sort(u2)
                zz = zz[u2]
                vv = vv[u2]
                if zz.size >= 3:
                    gg = np.gradient(vv, zz)
                    ax2.plot(gg, zz, linestyle="--", color=color, linewidth=1.5)
                    grad_plotted = True

        if (not grad_plotted) and (grad is not None) and (zmid is not None):
            ax2.plot(grad, zmid, linestyle="--", color=color, linewidth=1.5)

        ax2.set_xlabel(xlabel_top)

        if add_sg_thresh:
            ax2.axvline(SG_THRES, linestyle="-.", linewidth=1.2, color="0.3")

        if np.isfinite(core_depth):
            ax.axhline(core_depth, linestyle=":", linewidth=1.0, color="0.3")

        return ax2

    # T (red)
    axT = fig.add_subplot(gs[0, 0])
    panel(axT, SCHEME.centers, Tbin, Z_MID, TG,
          "T (°C)", "dT/dz (°C m$^{-1}$)",
          f"T profile | {meta['period']} {meta['season']}", "red",
          z_raw, T_raw, add_sg_thresh=False)

    # S (blue) + SG threshold line + annotation
    axS = fig.add_subplot(gs[0, 1])
    panel(axS, SCHEME.centers, Sbin, Z_MID, SG,
          "S (PSU)", "dS/dz (PSU m$^{-1}$)",
          f"S | SG_thres={SG_THRES:g}", "blue",
          z_raw, S_raw, add_sg_thresh=True)

    if np.isfinite(meta.get("z_core", np.nan)):
        axS.text(
            0.02, 0.02,
            f"zSG={meta['z_core']:.1f} m\nSG={meta.get('SG_core', np.nan):.3f}",
            transform=axS.transAxes,
            va="bottom", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.0)
        )

    # PDA (black)
    axD = fig.add_subplot(gs[0, 2])
    panel(axD, SCHEME.centers, Dbin, Z_MID, DG,
          "PDA (σ0, kg m$^{-3}$)", "dPDA/dz (kg m$^{-4}$)",
          "PDA", "black",
          z_raw, D_raw, add_sg_thresh=False)

    # Map (4th column; Cartopy is safe here, inset was not)
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
        axM.set_xlabel("lon")
        axM.set_ylabel("lat")
        axM.set_title("Location")
        if np.isfinite(lon) and np.isfinite(lat):
            axM.scatter([lon], [lat], s=35)
        axM.set_aspect("equal", adjustable="box")

    # Compact suptitle (avoid overlap)
    date_str = meta.get("date_str", "")
    cast_id = str(meta.get("cast_id", ""))
    cast_short = cast_id[:40] + ("…" if len(cast_id) > 40 else "")
    has_ssl = meta.get("has_ssl", 0)
    #fig.suptitle(
    #    f"{date_str} | {cast_short} |",
    #    y=0.98, fontsize=8
    #)
    fig.tight_layout(rect=[0, 0, 1, 0.82])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "profiles" / "20c").mkdir(parents=True, exist_ok=True)
    (OUTDIR / "profiles" / "21c").mkdir(parents=True, exist_ok=True)

    cast_rows = []

    for spec in SOURCES:
        if not spec.path.exists():
            print(f"[WARN] missing {spec.path} (skip)")
            continue

        print(f"[{spec.name}] reading {spec.path}")
        df = pd.read_csv(spec.path, low_memory=False)

        # basin filter
        basin = df[spec.basin_col].astype(str).str.strip()
        df = df[basin == TARGET_BASIN].copy()
        if df.empty:
            print("  -> no rows in this basin")
            continue

        # time + season + period
        df["_time"] = pd.to_datetime(df[spec.time_col], errors="coerce")
        df = df[pd.notna(df["_time"])].copy()
        df["_year"] = df["_time"].dt.year.astype(int)
        df["_month"] = df["_time"].dt.month.astype(int)
        df["_season"] = df["_month"].map(lambda m: month_to_season(int(m)))
        df = df[df["_season"].notna()].copy()
        df["_period"] = np.where(df["_year"] < 2000, "20c", "21c")

        # quick debug: rows distribution after basin+time filter
        print(f"  rows after basin/time filter: {len(df):,}")

        # group key: build a robust composite key from available cast_id_cols (e.g., Argo PLATFORM+ CYCLE)
        use_cols = [c for c in spec.cast_id_cols if c in df.columns]
        if use_cols:
            # composite key across all provided id columns (keeps cycles separate for Argo)
            df["_gkey"] = df[use_cols].astype(str).agg("|".join, axis=1)
            gkey = "_gkey"
        else:
            # fallback to time+lon+lat
            lon = pd.to_numeric(df.get(spec.lon_col), errors="coerce")
            lat = pd.to_numeric(df.get(spec.lat_col), errors="coerce")
            df["_gkey"] = df["_time"].dt.strftime("%Y-%m-%dT%H:%M") + "|" + lon.round(4).astype(str) + "|" + lat.round(4).astype(str)
            gkey = "_gkey"

        plotted = {"20c": 0, "21c": 0}

        for _, g in df.groupby(gkey, sort=False):
            cast_id = build_cast_id(g, spec)
            t0 = pd.to_datetime(g["_time"].iloc[0])
            season = g["_season"].iloc[0]
            period = g["_period"].iloc[0]
            lon = robust_first_valid(g, spec.lon_col)
            lat = robust_first_valid(g, spec.lat_col)

            Tbin, Sbin, Dbin, ok, z_raw, T_raw, S_raw, D_raw = binned_cast_arrays(g, spec)
            if not ok:
                continue

            TG, SG, DG = gradients_from_bins(Tbin, Sbin, Dbin)
            # SSL detection: use RAW metric with original-like conditions
            idx_core_raw, z_core, SG_core, TG_core, DG_core, SSI = ssl_core_from_raw(z_raw, S_raw, T_raw, D_raw)
            has_ssl = int(idx_core_raw >= 0)

            cast_rows.append(dict(
                source=spec.name,
                cast_id=cast_id,
                time=t0,
                season=season,
                period=period,
                lon=lon,
                lat=lat,
                has_ssl=has_ssl,
                z_core=z_core,
                SG_core=SG_core,
                TG_core=TG_core,
                DG_core=DG_core,
                SSI=SSI,
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
                lon=lon,
                lat=lat,
                has_ssl=has_ssl,
                z_core=z_core,
                SG_core=SG_core,
            )
            safe_id = cast_id.replace("/", "_").replace(" ", "_")
            out_png = OUTDIR / "profiles" / period / f"{t0.strftime('%Y%m%dT%H%M')}_{safe_id[:120]}.png"
            plot_cast_figure(out_png, meta, Tbin, Sbin, Dbin, TG, SG, DG, -1, z_raw, T_raw, S_raw, D_raw)
            plotted[period] += 1

        print(f"  -> plotted: {plotted}")

    # Save cast summary
    cast_df = pd.DataFrame(cast_rows)
    if cast_df.empty:
        print("\n[ERROR] No casts collected after filtering. Check basin codes and column names.")
        return
    cast_df.to_csv(OUTDIR / "cast_summary.csv", index=False)
    print("Wrote:", OUTDIR / "cast_summary.csv")

    # Summary counts + % SSL per season
    seasons_order = ["JFM", "AMJ", "JAS", "OND"]
    x = np.arange(len(seasons_order))
    width = 0.35

    grp = cast_df.groupby(["period", "season"]).size().reset_index(name="n_casts")
    c20 = [int(grp[(grp.period=="20c") & (grp.season==s)]["n_casts"].values[0]) if not grp[(grp.period=="20c") & (grp.season==s)].empty else 0 for s in seasons_order]
    c21 = [int(grp[(grp.period=="21c") & (grp.season==s)]["n_casts"].values[0]) if not grp[(grp.period=="21c") & (grp.season==s)].empty else 0 for s in seasons_order]

    plt.figure()
    plt.bar(x - width/2, c20, width, label="20c")
    plt.bar(x + width/2, c21, width, label="21c")
    plt.xticks(x, seasons_order)
    plt.ylabel("Number of casts")
    plt.title(f"{TARGET_BASIN}: sample size per season (all casts)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "summary_counts.png", dpi=150)
    plt.close()

    pct20, pct21 = [], []
    for s in seasons_order:
        sub20 = cast_df[(cast_df.period=="20c") & (cast_df.season==s)]
        sub21 = cast_df[(cast_df.period=="21c") & (cast_df.season==s)]
        pct20.append(100.0 * sub20["has_ssl"].mean() if len(sub20) else np.nan)
        pct21.append(100.0 * sub21["has_ssl"].mean() if len(sub21) else np.nan)

    plt.figure()
    plt.bar(x - width/2, pct20, width, label="20c")
    plt.bar(x + width/2, pct21, width, label="21c")
    plt.xticks(x, seasons_order)
    plt.ylabel("SSL percentage (%)")
    plt.title(f"{TARGET_BASIN}: SSL occurrence (SG<{SG_THRES:g}, z<={SSL_ZMAX:g}m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "summary_pct_ssl.png", dpi=150)
    plt.close()

    # Boxplots for SSL-only casts
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
        plt.xticks(range(1, len(labels)+1), labels)
        plt.ylabel(ylabel)
        plt.title(f"{TARGET_BASIN}: {var} (SSL casts only)")
        plt.tight_layout()
        plt.savefig(OUTDIR / outname, dpi=150)
        plt.close()

    if not ssl.empty:
        seasonal_boxplot("z_core", "zSG (m)", "box_zSG.png")
        seasonal_boxplot("SG_core", "SG_core (PSU m$^{-1}$)", "box_SG.png")
        seasonal_boxplot("TG_core", "TG_core (°C m$^{-1}$)", "box_TG.png")
        seasonal_boxplot("DG_core", "DG_core (kg m$^{-4}$)", "box_DG.png")

    print("Done. Outputs in:", OUTDIR.resolve())


if __name__ == "__main__":
    main()
