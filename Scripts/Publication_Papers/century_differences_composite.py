#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 2 prototype for Nature-style paper

Layout
------
Row 1: Mediterranean-wide Δ(21c-20c) maps at ~200 m
       a) ΔT   b) ΔS   c) ΔSIG0

Row 2: Vertical Δ profiles for representative composed basins
       d) NWM  e) ION  f) LEV
       Each basin block contains 3 narrow sub-axes:
           ΔT(z), ΔS(z), ΔSIG0(z)

Row 3: Water-mass interpretation
       g) ION T-S
       h) LEV T-S
       i) NWM T-S

Notes
-----
- Maps use the working deseasoned-monthly overall-delta logic from
  plot_manca_delta_overall_maps.py
- Vertical profiles use composed climatology CSVs (21c - 20c)
- T-S uses raw points below seasonal thermocline (z >= 100 m)
- First goal: get a working Figure 2 skeleton that you can refine visually
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from matplotlib.ticker import FormatStrFormatter

# Optional
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False

try:
    from shapely.geometry import Polygon
    HAVE_SHAPELY = True
except Exception:
    HAVE_SHAPELY = False

try:
    import gsw
    HAVE_GSW = True
except Exception:
    HAVE_GSW = False

from scipy.optimize import curve_fit


# ==========================================================
# USER SETTINGS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- project paths ---
MANCA_PY = str(PROJECT_ROOT.parents[1] / "Basins" / "Basins" / "manca.py")

# --- map data: same input as working delta maps ---
MONTHLY_TIMESERIES_CSV = str(PROJECT_ROOT / "Results/Trends/monthly_timeseries_all_bins.csv")

# --- climatology CSVs: composed basins ---
CLIM_21C_COMPOSED_CSV = str(PROJECT_ROOT / "Stats/MultiSource_Climatology_BINNED_ELIG/21c_2000_2025/POOLED_RAW/climatology_POOLED_RAW_21c_2000_2025_composed_after_sigma.csv")
CLIM_20C_COMPOSED_CSV = str(PROJECT_ROOT / "Stats/MultiSource_Climatology_BINNED_ELIG/20c_1950_1999/POOLED_RAW/climatology_POOLED_RAW_20c_1950_1999_composed_after_sigma.csv")

# --- raw profile sources (same philosophy as T-S script) ---

# --- output ---
OUTDIR = str(PROJECT_ROOT / "Plots/Paper_Figures")
OUTPNG = os.path.join(OUTDIR, "Figure2_composite_maps_profiles_TS.png")
OUTPDF = os.path.join(OUTDIR, "Figure2_composite_maps_profiles_TS.pdf")

# --- periods ---
Y20A, Y20B = 1950, 1999
Y21A, Y21B = 2000, 2025

# --- representative basins ---
BASIN_ORDER = ["NWM", "ION", "LEV"]

PROFILE_BASINS = BASIN_ORDER
TS_BASINS = [b.lower() for b in BASIN_ORDER]

# --- seasons ---
SEASON_ORDER = ["JFM", "AMJ", "JAS", "OND"]
SEASON_LABEL = {"JFM": "Winter", "AMJ": "Spring", "JAS": "Summer", "OND": "Autumn"}
SEASON_COLOR = {"JFM": "C0", "AMJ": "C2", "JAS": "C3", "OND": "C1"}

# --- map settings ---
MAP_EXTENT = (-6.5, 36.5, 30.0, 46.5)
SURF_RANGE = (0.0, 10.0)
DEPTH_TOL = 1.0
DEPTHS = [200.0, 1000.0, 2000.0]
DEPTH_TAGS = ["0-10"] + [str(int(d)) for d in DEPTHS]
DEPTH_TAG_MAP = "200"

MIN_N_TOTAL = 30
MIN_MONTHS = 24

# --- T-S settings ---
DEPTH_MIN_TS = 100.0
DEPTH_MAX_TS = 2000.0
MAX_POINTS_PER_REGION_PERIOD = None
RANDOM_SEED = 7
SCATTER_S = 1
SCATTER_ALPHA = 0.06

# --- profile settings ---
PROFILE_MAX_DEPTH = 2000.0

# Add automatic limits of your profiles
AUTO_PROFILE_LIMITS = True
PROFILE_LIMIT_DEPTH_MIN = 0.0   
PROFILE_LIMIT_QLOW = 2.0
PROFILE_LIMIT_QHIGH = 99.5
PROFILE_LIMIT_PAD = 1.12
PROFILE_LIMIT_MIN = {
    "dT": 0.15,
    "dS": 0.03,
    "dSIG0": 0.01,
}

# --- fonts / style ---
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
})

FIG_FACE = "white"
AX_FACE = "#f2f2f2"
LAND_COLOR = "#d9d6cf"
OCEAN_COLOR = "#f8f8fa"

MAP_CMAP = plt.get_cmap("RdBu_r")
DELTA_T_LIM = (-0.8, 0.8)
DELTA_S_LIM = (-0.20, 0.20)
DELTA_SIG0_LIM = (-0.08, 0.08)

PROFILE_T_LIM = (-1.0, 1.0)
PROFILE_S_LIM = (-0.26, 0.26)
PROFILE_SIG0_LIM = (-0.15, 0.15)

NO_DATA_FC = (0.65, 0.65, 0.65, 1.0)
NO_DATA_HATCH = "///"


# ==========================================================
# IMPORT THE CLIMATOLOGY SOURCE CONFIG
# ==========================================================

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from Scripts.MultiPlatform.core import multi_source_seasonal_climatology as msc
except Exception as e:
    raise ImportError(
        "Could not import Scripts.MultiPlatform.core.multi_source_seasonal_climatology "
        f"from {PROJECT_ROOT}"
    ) from e


# ==========================================================
# HELPERS: loading Manca polygons
# ==========================================================

def parse_manca_py_rectangles(manca_py_path: str):
    """
    Robust parser for Basin.from_array(...) in manca.py.
    Handles both np.array([[...],[...]]) and np.array([ [...], ... ]) styles.
    """
    if not HAVE_SHAPELY:
        raise ImportError("shapely is required for parsing Manca polygons.")

    txt = open(manca_py_path, "r", encoding="utf-8", errors="ignore").read()

    pat = re.compile(
        r"Basin\.from_array\(\s*'(?P<code>[A-Z0-9]+)'\s*,.*?np\.array\(\s*(?P<arr>\[[\s\S]*?\])\s*\)\s*\)",
        re.DOTALL
    )
    triple_pat = re.compile(
        r"\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*0\.?0?\s*\]"
    )

    basins = {}
    for m in pat.finditer(txt):
        code = m.group("code").strip().upper()
        coords = [(float(lon), float(lat)) for lon, lat in triple_pat.findall(m.group("arr"))]
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        basins[code] = Polygon(coords)

    if not basins:
        raise RuntimeError(f"No basins parsed from {manca_py_path}.")
    return basins


# ==========================================================
# HELPERS: monthly-map delta
# exact working logic from plot_manca_delta_overall_maps.py
# ==========================================================

def load_monthly_timeseries(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["time"] = pd.to_datetime(df["time"])

    need = {"basin", "z_m", "var", "time", "year", "month", "value", "n"}
    miss = need.difference(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in {path}: {sorted(miss)}")

    df["basin"] = df["basin"].astype(str).str.strip().str.upper()
    df["var"] = df["var"].astype(str).str.strip().str.upper()
    df["z_m"] = pd.to_numeric(df["z_m"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")

    return df.dropna(subset=["time", "basin", "var", "z_m", "value", "n", "year", "month"]).copy()


def weighted_mean(values, weights) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.sum(v[m] * w[m]) / np.sum(w[m]))


def pool_monthly(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    If there are multiple entries per month, pool them weighted by n.
    Returns: time, year, month, value, w
    """
    if df_sub.empty:
        return df_sub

    rows = []
    for (time, year, month), g in df_sub.groupby(["time", "year", "month"]):
        rows.append({
            "time": time,
            "year": year,
            "month": month,
            "value": weighted_mean(g["value"].to_numpy(), g["n"].to_numpy()),
            "w": float(np.nansum(g["n"].to_numpy()))
        })

    out = pd.DataFrame(rows)
    return out.sort_values("time").reset_index(drop=True)


def pick_series_for_depth(df_bv: pd.DataFrame, depth_tag: str) -> pd.DataFrame:
    """
    For a single basin+var monthly dataset:
      - "0-10": pool all z in SURF_RANGE per month
      - numeric depth: pick nearest z within DEPTH_TOL, pool per month
    Returns dataframe with time/year/month/value/w.
    """
    if depth_tag == "0-10":
        d = df_bv[(df_bv["z_m"] >= SURF_RANGE[0]) & (df_bv["z_m"] <= SURF_RANGE[1])].copy()
        return pool_monthly(d)

    target = float(depth_tag)
    z = df_bv["z_m"].to_numpy(float)
    if z.size == 0:
        return df_bv.iloc[0:0].copy()

    uniq_z = np.unique(z[np.isfinite(z)])
    if uniq_z.size == 0:
        return df_bv.iloc[0:0].copy()

    nearest = float(uniq_z[np.argmin(np.abs(uniq_z - target))])
    if abs(nearest - target) > DEPTH_TOL:
        return df_bv.iloc[0:0].copy()

    d = df_bv[np.isclose(df_bv["z_m"].to_numpy(float), nearest)].copy()
    return pool_monthly(d)


def t_months_since_start(times: pd.Series) -> np.ndarray:
    times = pd.to_datetime(times)
    t0 = times.min()
    return ((times - t0) / np.timedelta64(1, "D")).to_numpy(float) / 30.4375


def seasonal_cos_model(t_months: np.ndarray, A: float, B: float, phi1: float, phi2: float, Y: float) -> np.ndarray:
    t = np.asarray(t_months, float)
    return (
        A * np.cos(2.0 * np.pi * t / 12.0 + phi1) +
        B * np.cos(2.0 * np.pi * t / 6.0 + phi2) +
        Y
    )


def deseason_like_original(times: pd.Series, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit seasonal cosine model (12 & 6 months) and return:
      y_fit_full, y_deseas_full, Y_hat
    where y_deseas = y - y_fit  (mean removed!), Y_hat is fitted mean.
    """
    times = pd.to_datetime(times)
    y = np.asarray(y, float)

    t_full = t_months_since_start(times)
    m = np.isfinite(t_full) & np.isfinite(y)

    y_fit_full = np.full_like(y, np.nan, dtype=float)
    y_deseas_full = np.full_like(y, np.nan, dtype=float)

    if m.sum() < 12:
        return y_fit_full, y_deseas_full, np.nan

    t = t_full[m]
    yy = y[m]

    p0 = [1.0, 1.0, 0.0, 0.0, float(np.nanmean(yy))]
    bounds = ([-np.inf, -np.inf, -np.pi, -np.pi, -np.inf],
              [ np.inf,  np.inf,  np.pi,  np.pi,  np.inf])

    try:
        params, _ = curve_fit(seasonal_cos_model, t, yy, p0=p0, bounds=bounds, maxfev=20000)
        fit = seasonal_cos_model(t, *params)
        y_fit_full[m] = fit
        y_deseas_full[m] = yy - fit
        Y_hat = float(params[4])
        return y_fit_full, y_deseas_full, Y_hat
    except Exception:
        return y_fit_full, y_deseas_full, np.nan


def mean_preserving_deseason(times: pd.Series, y: np.ndarray) -> np.ndarray:
    """
    Remove only the seasonal part, keep mean:
      y_adj = y - (y_fit - Y_hat) = (y - y_fit) + Y_hat
    """
    y_fit, y_deseas, Y_hat = deseason_like_original(times, y)
    if not np.isfinite(Y_hat):
        return np.full_like(np.asarray(y, float), np.nan, dtype=float)
    return y_deseas + Y_hat


def compute_century_mean_from_series(s: pd.DataFrame, y_adj: np.ndarray, y0: int, y1: int) -> tuple[float, float, int]:
    """
    s columns: year, w ; y_adj aligned to s rows.
    Returns: (mean, n_total, n_months)
    """
    if s.empty:
        return (np.nan, 0.0, 0)

    m = (
        (s["year"].to_numpy(int) >= y0) &
        (s["year"].to_numpy(int) <= y1) &
        np.isfinite(y_adj) &
        np.isfinite(s["w"].to_numpy(float))
    )
    if not np.any(m):
        return (np.nan, 0.0, 0)

    yy = np.asarray(y_adj, float)[m]
    ww = s["w"].to_numpy(float)[m]
    mean = weighted_mean(yy, ww)
    n_total = float(np.nansum(ww))
    n_months = int(np.sum(m))
    return (mean, n_total, n_months)


def build_overall_delta_from_monthly(df_monthly: pd.DataFrame, var: str) -> pd.DataFrame:
    """
    Returns:
      basin, depth_tag, delta, mean20, mean21, n20, n21, m20, m21, status
    """
    out_rows = []
    dfv = df_monthly[df_monthly["var"] == var].copy()
    if dfv.empty:
        return pd.DataFrame(columns=["basin", "depth_tag", "delta", "mean20", "mean21", "n20", "n21", "m20", "m21", "status"])

    for basin, g_b in dfv.groupby("basin"):
        for depth_tag in DEPTH_TAGS:
            s = pick_series_for_depth(g_b, depth_tag)
            if s.empty:
                out_rows.append((basin, depth_tag, np.nan, np.nan, np.nan, 0.0, 0.0, 0, 0, "no_data"))
                continue

            y_adj = mean_preserving_deseason(s["time"], s["value"].to_numpy(float))

            mean20, n20, m20 = compute_century_mean_from_series(s, y_adj, Y20A, Y20B)
            mean21, n21, m21 = compute_century_mean_from_series(s, y_adj, Y21A, Y21B)

            ok = (
                np.isfinite(mean20) and np.isfinite(mean21) and
                (n20 >= MIN_N_TOTAL) and (n21 >= MIN_N_TOTAL) and
                (m20 >= MIN_MONTHS) and (m21 >= MIN_MONTHS)
            )

            delta = (mean21 - mean20) if ok else np.nan
            status = "ok" if np.isfinite(delta) else "no_data"
            out_rows.append((basin, depth_tag, delta, mean20, mean21, n20, n21, m20, m21, status))

    return pd.DataFrame(
        out_rows,
        columns=["basin", "depth_tag", "delta", "mean20", "mean21", "n20", "n21", "m20", "m21", "status"]
    )


# ==========================================================
# HELPERS: climatology / vertical profiles
# ==========================================================

def load_climatology(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    if "basin" in df.columns and "BASIN" not in df.columns:
        df = df.rename(columns={"basin": "BASIN"})
    if "sig0_mean" in df.columns and "SIG0_mean" not in df.columns:
        df = df.rename(columns={"sig0_mean": "SIG0_mean"})
    if "sig0_std" in df.columns and "SIG0_std" not in df.columns:
        df = df.rename(columns={"sig0_std": "SIG0_std"})

    req = ["BASIN", "season", "z_m", "T_mean", "S_mean", "SIG0_mean"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in climatology file {path}: {missing}")

    out = df.copy()
    out["BASIN"] = out["BASIN"].astype(str).str.strip().str.upper()
    out["season"] = out["season"].astype(str).str.strip().str.upper()
    out["z_m"] = pd.to_numeric(out["z_m"], errors="coerce")
    for c in ["T_mean", "S_mean", "SIG0_mean"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_profile_delta(clim21: pd.DataFrame, clim20: pd.DataFrame, basin: str) -> pd.DataFrame:
    b = basin.upper().strip()
    d21 = clim21[clim21["BASIN"] == b].copy()
    d20 = clim20[clim20["BASIN"] == b].copy()

    merged = d21.merge(
        d20,
        on=["BASIN", "season", "z_m"],
        how="inner",
        suffixes=("_21", "_20")
    )

    merged["dT"] = merged["T_mean_21"] - merged["T_mean_20"]
    merged["dS"] = merged["S_mean_21"] - merged["S_mean_20"]
    merged["dSIG0"] = merged["SIG0_mean_21"] - merged["SIG0_mean_20"]

    merged = merged.sort_values(["season", "z_m"]).copy()
    return merged


# ==========================================================
# HELPERS: T-S
# ==========================================================

COL_CANDIDATES = {
    "time": [
        "TIME", "time", "date", "datetime", "DATE", "Date",
        "yyyy-mm-ddThh:mm:ss.sss",
    ],
    "depth": ["DEPTH_M", "depth_m", "depth", "Depth [m]", "DEPTH", "PRES"],
    "temp": [
        "TEMP_C", "temp_c", "temperature", "temp", "TEMP",
        "ITS-90 water temperature [degrees C]"
    ],
    "sal": [
        "SAL_PSU", "sal_psu", "salinity", "sal", "PSAL",
        "Water body salinity [per mille]"
    ],
    "basin": ["basin", "BASIN", "BASIN_3ADR_FIXED", "BASIN_3ADR", "basin_id", "manca_basin"],
    "lon": ["lon", "longitude", "LONGITUDE", "Longitude [degrees_east]"],
    "lat": ["lat", "latitude", "LATITUDE", "Latitude [degrees_north]"],
}

AXIS_LIMITS = {
    "ion": {"S": (38.3, 39.2), "T": (13.3, 17.6)},
    "lev": {"S": (38.6, 39.3), "T": (13.5, 18.1)},
    "nwm": {"S": (38.1, 38.7), "T": (12.8, 14.4)},
    "alb": {"S": (38.0, 38.6), "T": (12.8, 14.0)},
}


def pick_col_from_list(columns: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def detect_columns_in_file(fp: str) -> dict[str, str | None]:
    cols = pd.read_csv(fp, nrows=0).columns.tolist()
    return {
        "time":  pick_col_from_list(cols, COL_CANDIDATES["time"]),
        "depth": pick_col_from_list(cols, COL_CANDIDATES["depth"]),
        "temp":  pick_col_from_list(cols, COL_CANDIDATES["temp"]),
        "sal":   pick_col_from_list(cols, COL_CANDIDATES["sal"]),
        "basin": pick_col_from_list(cols, COL_CANDIDATES["basin"]),
        "lon":   pick_col_from_list(cols, COL_CANDIDATES["lon"]),
        "lat":   pick_col_from_list(cols, COL_CANDIDATES["lat"]),
    }


def build_usecols_and_dtypes(colmap: dict[str, str | None]) -> tuple[list[str], dict[str, str]]:
    required = ["time", "depth", "temp", "sal", "basin"]
    missing = [k for k in required if colmap[k] is None]
    if missing:
        raise ValueError(f"Missing required columns {missing} in file.")

    usecols = [v for v in colmap.values() if v is not None]

    dtype_map: dict[str, str] = {}
    if colmap["depth"] is not None:
        dtype_map[colmap["depth"]] = "float32"
    if colmap["temp"] is not None:
        dtype_map[colmap["temp"]] = "float32"
    if colmap["sal"] is not None:
        dtype_map[colmap["sal"]] = "float32"
    if colmap["basin"] is not None:
        dtype_map[colmap["basin"]] = "string"
    if colmap["lon"] is not None:
        dtype_map[colmap["lon"]] = "float32"
    if colmap["lat"] is not None:
        dtype_map[colmap["lat"]] = "float32"

    return usecols, dtype_map


def read_one_raw_source(fp: str) -> pd.DataFrame:
    colmap = detect_columns_in_file(fp)
    usecols, dtype_map = build_usecols_and_dtypes(colmap)
    df = pd.read_csv(fp, usecols=usecols, dtype=dtype_map, low_memory=False)

    out = pd.DataFrame({
        "time_raw": df[colmap["time"]],
        "depth_m": df[colmap["depth"]],
        "temp_c": df[colmap["temp"]],
        "sal_psu": df[colmap["sal"]],
        "basin_atomic": df[colmap["basin"]],
        "lon": df[colmap["lon"]] if colmap["lon"] else np.nan,
        "lat": df[colmap["lat"]] if colmap["lat"] else np.nan,
    })

    out["year"] = pd.to_numeric(
        out["time_raw"].astype(str).str.slice(0, 4),
        errors="coerce"
    ).astype("float32")

    out = out.drop(columns=["time_raw"])

    out["depth_m"] = pd.to_numeric(out["depth_m"], errors="coerce").astype("float32")
    out["temp_c"] = pd.to_numeric(out["temp_c"], errors="coerce").astype("float32")
    out["sal_psu"] = pd.to_numeric(out["sal_psu"], errors="coerce").astype("float32")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce").astype("float32")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce").astype("float32")

    out["basin_atomic"] = (
        out["basin_atomic"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    return out


def build_atomic_to_composed_map(comp: dict) -> dict[str, str]:
    m = {}
    for comp_id, info in comp.items():
        for b in info.get("members", []):
            m[str(b).strip().upper()] = str(comp_id).strip().lower()
    return m


def subsample(df: pd.DataFrame, nmax: int | None, seed: int) -> pd.DataFrame:
    if (nmax is None) or (len(df) <= nmax):
        return df
    return df.sample(n=nmax, random_state=seed)


def load_ts_dataframe() -> pd.DataFrame:
    input_csvs = [str(src.path) for src in msc.SOURCES]
    atomic2comp = build_atomic_to_composed_map(msc.COMPOSED_BASINS)

    parts = []
    for fp in input_csvs:
        if not os.path.exists(fp):
            print(f"[WARN] missing source: {fp}")
            continue
        print(f"[INFO] reading T-S source: {fp}")
        d = read_one_raw_source(fp)
        d["region"] = d["basin_atomic"].map(atomic2comp)
        parts.append(d)

    if not parts:
        raise RuntimeError("No raw source data loaded for T-S plots.")

    df = pd.concat(parts, ignore_index=True)

    df = df[
        np.isfinite(df["year"]) &
        np.isfinite(df["depth_m"]) &
        np.isfinite(df["temp_c"]) &
        np.isfinite(df["sal_psu"]) &
        df["region"].notna()
    ].copy()

    df = df[(df["depth_m"] >= DEPTH_MIN_TS)]
    if DEPTH_MAX_TS is not None:
        df = df[(df["depth_m"] <= DEPTH_MAX_TS)].copy()

    return df


def compute_sigma0_grid(SP_grid, t_grid, lon=None, lat=None):
    if not HAVE_GSW:
        return None

    p0 = np.zeros_like(SP_grid)

    try:
        if lon is not None and lat is not None and np.isfinite(lon) and np.isfinite(lat):
            SA = gsw.SA_from_SP(SP_grid, p0, lon, lat)
            CT = gsw.CT_from_t(SA, t_grid, p0)
            return gsw.sigma0(SA, CT)
        return gsw.sigma0(SP_grid, t_grid)
    except Exception:
        return None


def add_pda_isolines(ax, d_for_loc: pd.DataFrame | None):
    if not HAVE_GSW:
        return

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    S = np.linspace(x0, x1, 140)
    T = np.linspace(y0, y1, 140)
    SPg, Tg = np.meshgrid(S, T)

    lon = lat = None
    if d_for_loc is not None and len(d_for_loc):
        lon_med = pd.to_numeric(d_for_loc["lon"], errors="coerce").median()
        lat_med = pd.to_numeric(d_for_loc["lat"], errors="coerce").median()
        if np.isfinite(lon_med) and np.isfinite(lat_med):
            lon, lat = float(lon_med), float(lat_med)

    sig0 = compute_sigma0_grid(SPg, Tg, lon=lon, lat=lat)
    if sig0 is None:
        return

    levels = np.arange(27.0, 30.5, 0.2)
    cs = ax.contour(
        SPg, Tg, sig0,
        levels=levels,
        colors="0.35",
        linewidths=0.8,
        alpha=0.45,
        zorder=1,
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")


# ==========================================================
# PLOTTING HELPERS
# ==========================================================

def add_panel_label(ax, label: str):
    ax.text(
        0.01, 0.99, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13, fontweight="bold"
    )


def setup_geo_ax(ax):
    if HAVE_CARTOPY:
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, edgecolor="none", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor="none", zorder=5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=6)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_facecolor(AX_FACE)
        ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
        ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
        ax.set_xticks([])
        ax.set_yticks([])


def plot_delta_map(ax, basin_polys, delta_df, depth_tag, vlim, title, label):
    setup_geo_ax(ax)
    add_panel_label(ax, label)

    dsub = delta_df[delta_df["depth_tag"] == depth_tag].copy()
    lut = dict(zip(dsub["basin"], dsub["delta"]))

    vmin, vmax = vlim
    cmap = MAP_CMAP

    for code, poly in basin_polys.items():
        val = lut.get(code.upper(), np.nan)
        if np.isfinite(val):
            fc = cmap((val - vmin) / (vmax - vmin))
            hatch = None
        else:
            fc = NO_DATA_FC
            hatch = NO_DATA_HATCH

        if HAVE_CARTOPY:
            ax.add_geometries(
                [poly], ccrs.PlateCarree(),
                facecolor=fc, edgecolor="k", linewidth=0.25, hatch=hatch, zorder=2
            )
        else:
            x, y = poly.exterior.xy
            ax.fill(x, y, facecolor=fc, edgecolor="k", linewidth=0.25, hatch=hatch)

    ax.set_title(title, pad=4)

def sparse_symmetric_ticks(xmin: float, xmax: float, var: str):
    half = max(abs(xmin), abs(xmax))

    if var == "dT":
        if half <= 0.3:
            nice = 0.2
        elif half <= 0.6:
            nice = 0.5
        elif half <= 1.2:
            nice = 1.0
        else:
            nice = 1.5

    elif var == "dS":
        if half <= 0.06:
            nice = 0.05
        elif half <= 0.12:
            nice = 0.10
        elif half <= 0.22:
            nice = 0.20
        else:
            nice = 0.25

    elif var == "dSIG0":
        if half <= 0.03:
            nice = 0.03
        elif half <= 0.06:
            nice = 0.05
        elif half <= 0.10:
            nice = 0.08
        else:
            nice = 0.10

    else:
        nice = half

    return [-nice, 0.0, nice]

def plot_profile_block(
    fig,
    subspec,
    basin,
    delta_df,
    xlims,
    label=None,
    show_ylabel=False
    ):
    """
    One basin block = 3 narrow axes sharing depth:
      [ΔT] [ΔS] [ΔSIG0]
    """
    subgs = GridSpecFromSubplotSpec(1, 3, subplot_spec=subspec, wspace=0.22)

    axT = fig.add_subplot(subgs[0, 0])
    axS = fig.add_subplot(subgs[0, 1], sharey=axT)
    axD = fig.add_subplot(subgs[0, 2], sharey=axT)

    for ax in [axT, axS, axD]:
        ax.set_facecolor(AX_FACE)
        ax.grid(True, alpha=0.25)
        ax.axvline(0.0, color="0.25", lw=0.8)
        ax.invert_yaxis()
        ax.set_ylim(PROFILE_MAX_DEPTH, 0)

    if label is not None:
        add_panel_label(axT, label)

    axT.set_title(f"{basin}", pad=1)

    for season in SEASON_ORDER:
        d = delta_df[delta_df["season"] == season].sort_values("z_m")
        if d.empty:
            continue
        c = SEASON_COLOR[season]
        axT.plot(d["dT"], d["z_m"], color=c, lw=1.8, label=SEASON_LABEL[season])
        axS.plot(d["dS"], d["z_m"], color=c, lw=1.8)
        axD.plot(d["dSIG0"], d["z_m"], color=c, lw=1.8)

    axT.set_xlabel("ΔT [°C]")
    axS.set_xlabel("ΔS [-]")
    #axD.set_xlabel("Δσ₀ [kg m⁻³]")
    axD.set_xlabel("ΔPDA [kg m⁻³]")

    axT.set_xlim(*xlims["dT"])
    axS.set_xlim(*xlims["dS"])
    axD.set_xlim(*xlims["dSIG0"])

    axT.set_xticks(sparse_symmetric_ticks(*xlims["dT"], var="dT"))
    axS.set_xticks(sparse_symmetric_ticks(*xlims["dS"], var="dS"))
    axD.set_xticks(sparse_symmetric_ticks(*xlims["dSIG0"], var="dSIG0"))

    axT.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axS.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axD.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    axT.tick_params(axis="x", labelsize=8.5, pad=1)
    axS.tick_params(axis="x", labelsize=8.5, pad=1)
    axD.tick_params(axis="x", labelsize=8.5, pad=1)

    if show_ylabel:
        axT.set_ylabel("Depth [m]")
    else:
        axT.set_ylabel("")
        plt.setp(axT.get_yticklabels(), visible=False)

    plt.setp(axS.get_yticklabels(), visible=False)
    plt.setp(axD.get_yticklabels(), visible=False)

    return axT, axS, axD


def plot_ts_panel(ax, df_ts, region, label):
    ax.set_facecolor(AX_FACE)
    add_panel_label(ax, label)

    d = df_ts[df_ts["region"] == region].copy()
    d20 = d[(d["year"] >= Y20A) & (d["year"] <= Y20B)].copy()
    d21 = d[(d["year"] >= Y21A) & (d["year"] <= Y21B)].copy()

    d20 = subsample(d20, MAX_POINTS_PER_REGION_PERIOD, RANDOM_SEED)
    d21 = subsample(d21, MAX_POINTS_PER_REGION_PERIOD, RANDOM_SEED + 1)

    lims = AXIS_LIMITS[region]
    ax.set_xlim(*lims["S"])
    ax.set_ylim(*lims["T"])

    add_pda_isolines(ax, d)

    norm = Normalize(vmin=DEPTH_MIN_TS, vmax=DEPTH_MAX_TS if DEPTH_MAX_TS is not None else 2000.0)

    ax.scatter(
        d20["sal_psu"], d20["temp_c"],
        c=d20["depth_m"], cmap="Blues", norm=norm,
        s=SCATTER_S, alpha=SCATTER_ALPHA, linewidths=0, rasterized=True, zorder=2
    )
    ax.scatter(
        d21["sal_psu"], d21["temp_c"],
        c=d21["depth_m"], cmap="Reds", norm=norm,
        s=SCATTER_S, alpha=SCATTER_ALPHA, linewidths=0, rasterized=True, zorder=3
    )

    ax.set_xlabel("Salinity")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(region.upper())

    n20 = len(d20)
    n21 = len(d21)
    ax.text(
        0.98, 0.03, f"20c n={n20:,}\n21c n={n21:,}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.75", alpha=0.9)
    )

def panel_letter(idx: int) -> str:
    return chr(ord("a") + idx)


def compute_auto_profile_limits(profile_deltas: dict[str, pd.DataFrame]) -> dict[str, tuple[float, float]]:
    """
    Compute symmetric x-limits for dT, dS, dSIG0 from all selected basins,
    using robust percentiles in the upper 0-200 m layer.

    This keeps the surface signal visible, but avoids a few extreme values
    blowing up the whole axis range.
    """
    out = {}

    depth_min = 0.0
    depth_max = 200.0
    qlow = 2.0
    qhigh = 98.0
    pad = 1.08

    hard_caps = {
        "dT": 1.5,
        "dS": 0.25,
        "dSIG0": 0.10,
    }

    for var in ["dT", "dS", "dSIG0"]:
        vals_all = []

        for basin, df in profile_deltas.items():
            if var not in df.columns or "z_m" not in df.columns:
                continue

            d = df.copy()
            d["z_m"] = pd.to_numeric(d["z_m"], errors="coerce")
            d[var] = pd.to_numeric(d[var], errors="coerce")

            d = d[
                np.isfinite(d["z_m"]) &
                np.isfinite(d[var]) &
                (d["z_m"] >= depth_min) &
                (d["z_m"] <= depth_max)
            ]

            if not d.empty:
                vals_all.append(d[var].to_numpy(float))

        if not vals_all:
            half = PROFILE_LIMIT_MIN[var]
            out[var] = (-half, half)
            continue

        vals = np.concatenate(vals_all)
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            half = PROFILE_LIMIT_MIN[var]
            out[var] = (-half, half)
            continue

        ql = np.nanpercentile(vals, qlow)
        qh = np.nanpercentile(vals, qhigh)

        half = max(abs(ql), abs(qh))
        half = max(half * pad, PROFILE_LIMIT_MIN[var])
        half = min(half, hard_caps[var])

        out[var] = (-half, half)

    return out


# ==========================================================
# MAIN
# ==========================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("[INFO] Parsing Manca polygons")
    basin_polys = parse_manca_py_rectangles(MANCA_PY)
    print(f"[DEBUG] parsed polygons: {len(basin_polys)}")

    print("[INFO] Loading monthly timeseries")
    df_month = load_monthly_timeseries(MONTHLY_TIMESERIES_CSV)

    print("[INFO] Computing 200 m delta maps with working overall-delta logic")
    delta_T = build_overall_delta_from_monthly(df_month, "T")
    delta_S = build_overall_delta_from_monthly(df_month, "S")
    delta_D = build_overall_delta_from_monthly(df_month, "SIG0")

    print("[DEBUG] delta_T finite @200:", np.isfinite(delta_T.loc[delta_T["depth_tag"] == DEPTH_TAG_MAP, "delta"]).sum())
    print("[DEBUG] delta_S finite @200:", np.isfinite(delta_S.loc[delta_S["depth_tag"] == DEPTH_TAG_MAP, "delta"]).sum())
    print("[DEBUG] delta_D finite @200:", np.isfinite(delta_D.loc[delta_D["depth_tag"] == DEPTH_TAG_MAP, "delta"]).sum())

    print("[INFO] Loading climatologies")
    clim21 = load_climatology(CLIM_21C_COMPOSED_CSV)
    clim20 = load_climatology(CLIM_20C_COMPOSED_CSV)

    profile_deltas = {}
    for basin in PROFILE_BASINS:
        print(f"[INFO] Building vertical delta profiles: {basin}")
        profile_deltas[basin] = build_profile_delta(clim21, clim20, basin)

    if AUTO_PROFILE_LIMITS:
        profile_xlims = compute_auto_profile_limits(profile_deltas)
    else:
        profile_xlims = {
            "dT": PROFILE_T_LIM,
            "dS": PROFILE_S_LIM,
            "dSIG0": PROFILE_SIG0_LIM,
        }

    print("[DEBUG] profile_xlims:", profile_xlims)

    print("[INFO] Loading T-S dataframe")
    df_ts = load_ts_dataframe()

    print("[INFO] Creating figure")
    fig = plt.figure(figsize=(18.4, 11.0), facecolor=FIG_FACE)
    gs = GridSpec(
        3, 3, figure=fig,
        height_ratios=[0.96, 0.92, 1.12],
        width_ratios=[1.00, 1.00, 1.00],
        hspace=0.26, wspace=0.22
    )
    # ---------------- top row: maps ----------------
    if HAVE_CARTOPY:
        ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        ax_c = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
    else:
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])

    plot_delta_map(ax_a, basin_polys, delta_T, DEPTH_TAG_MAP, DELTA_T_LIM, "ΔT @ 200 m", "a")
    plot_delta_map(ax_b, basin_polys, delta_S, DEPTH_TAG_MAP, DELTA_S_LIM, "ΔS @ 200 m", "b")
    #plot_delta_map(ax_c, basin_polys, delta_D, DEPTH_TAG_MAP, DELTA_SIG0_LIM, "Δσ₀ @ 200 m", "c")
    plot_delta_map(ax_c, basin_polys, delta_D, DEPTH_TAG_MAP, DELTA_SIG0_LIM, "ΔPDA @ 200 m", "c")

    for ax, lim, unit, ticks in [
        (ax_a, DELTA_T_LIM, "ΔT [°C]", np.arange(-0.8, 0.81, 0.2)),
        (ax_b, DELTA_S_LIM, "ΔS [-]", np.arange(-0.20, 0.201, 0.05)),
        #(ax_c, DELTA_SIG0_LIM, "Δσ₀ [kg m⁻³]", np.arange(-0.08, 0.081, 0.02)),
        (ax_c, DELTA_SIG0_LIM, "ΔPDA [kg m⁻³]", np.arange(-0.08, 0.081, 0.02)),
    ]:
        cax = ax.inset_axes([0.08, -0.16, 0.84, 0.065])
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(*lim), cmap=MAP_CMAP)
        cb = plt.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks)
        cb.set_label(unit, fontsize=10)
        cb.ax.tick_params(labelsize=9, pad=2)

    # ---------------- middle row: profiles ----------------
    profile_axes = []

    for j, basin in enumerate(PROFILE_BASINS):
        axes_tuple = plot_profile_block(
            fig,
            gs[1, j],
            basin,
            profile_deltas[basin],
            xlims=profile_xlims,
            label=panel_letter(3 + j),
            show_ylabel=(j == 0)
        )
        profile_axes.append(axes_tuple)

    axT_last, axS_last, axD_last = profile_axes[-1]

    handles = [Line2D([0], [0], color=SEASON_COLOR[s], lw=2.5, label=SEASON_LABEL[s]) for s in SEASON_ORDER]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.65),
        ncol=4,
        frameon=False,
        fontsize=10,
        handlelength=2.5,
        columnspacing=1.5
    )

    # ---------------- bottom row: T-S ----------------
    ts_axes = []

    for j, basin_lc in enumerate(TS_BASINS):
        ax = fig.add_subplot(gs[2, j])
        plot_ts_panel(ax, df_ts, basin_lc, panel_letter(6 + j))   # g, h, i ...
        ts_axes.append(ax)

    ts_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="tab:blue",
               markeredgecolor="tab:blue", markersize=6, label="20th c."),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="tab:red",
               markeredgecolor="tab:red", markersize=6, label="21st c."),
    ]
    if len(ts_axes) >= 2:
        ts_axes[1].legend(handles=ts_handles, loc="upper left", frameon=False)
    elif len(ts_axes) >= 1:
        ts_axes[0].legend(handles=ts_handles, loc="upper left", frameon=False)

    fig.savefig(OUTPNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPDF, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote {OUTPNG}")
    print(f"[OK] Wrote {OUTPDF}")


if __name__ == "__main__":
    main()