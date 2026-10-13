#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reads : Data/Seadatanet/*_mainland_special_excluded_no_argo_copies.csv, Data/Argo/CSV/Argo_with_basins_mainland_special_excluded.csv, Data/Adriatic/Processed/*_long_fixed_mainland_special_excluded.csv (SOURCES of multi_source_seasonal_climatology.py)
# Writes: Plots/MultiSource_Climatology/TS_2x2_composed_regions_depthcoded_masked.png (+ _CACHE.parquet)

"""
2x2 T–S diagrams for composed regions (alb, nwm, ion, lev)
with (T,S) pairs below seasonal thermocline (default: z >= 100 m).

FAST VERSION
------------
Main speed-ups:
- read only needed columns from CSVs
- specify dtypes where possible
- extract only year (no full datetime parsing)
- cache processed dataframe to parquet
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ==========================================================
# Import the multi-source climatology config
# ==========================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Scripts.paths import PLOTS

try:
    from Scripts.MultiPlatform.core import multi_source_seasonal_climatology as msc
except ImportError as e:
    raise ImportError(
        "Could not import Scripts.MultiPlatform.core.multi_source_seasonal_climatology.\n"
        "Make sure you are running from within the MedClim repository."
    ) from e

# ==========================================================
# Embedded configuration
# ==========================================================

INPUT_CSVS = [str(src.path) for src in msc.SOURCES]

P20_START, P20_END = 1950, 1999
P21_START, P21_END = 2000, 2025

DEPTH_MIN = 100.0
DEPTH_MAX = 2000.0   # set to None to disable upper cap

REGIONS = ["ion", "lev", "nwm", "alb"]
COMPOSED_BASINS = msc.COMPOSED_BASINS
BASIN_COMPOSED_COL: str | None = None

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

MAX_POINTS_PER_REGION_PERIOD = None
RANDOM_SEED = 7

OUTDIR = str(PLOTS / "MultiSource_Climatology")
OUTPNG = os.path.join(OUTDIR, "TS_2x2_composed_regions_depthcoded_masked.png")
CACHE_PARQUET = os.path.join(OUTDIR, "TS_2x2_composed_regions_depthcoded_masked_CACHE.parquet")
DPI = 300

SCATTER_S = 1
SCATTER_ALPHA = 0.08

PLOT_DEPTH_MARKERS = False
DEPTH_MARKERS = [100, 150, 200, 300, 500, 700, 1000, 1500, 2000]
DEPTH_MARKER_HALF_WINDOW = 20
DEPTH_MARKER_MIN_N = 80

PLOT_PDA_ISOLINES = True
PDA_LEVELS = np.arange(27.0, 30.5, 0.2)
PDA_LINEWIDTH = 0.8
PDA_ALPHA = 0.45
PDA_LABELS = True

AXIS_LIMITS = {
    "ion": {"S": (38.3, 39.2), "T": (13.3, 17.6)},
    "lev": {"S": (38.6, 39.3), "T": (13.5, 18.1)},
    "nwm": {"S": (38.1, 38.7), "T": (12.8, 14.4)},
    "alb": {"S": (38.0, 38.6), "T": (12.8, 14.0)},
}

# ==========================================================
# Style
# ==========================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ==========================================================
# Helpers
# ==========================================================

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


def standardize_one(df: pd.DataFrame, colmap: dict[str, str | None]) -> pd.DataFrame:
    c_time = colmap["time"]
    c_z = colmap["depth"]
    c_T = colmap["temp"]
    c_S = colmap["sal"]
    c_basin = colmap["basin"]
    c_lon = colmap["lon"]
    c_lat = colmap["lat"]

    out = pd.DataFrame({
        "time_raw": df[c_time],
        "depth_m": df[c_z],
        "temp_c": df[c_T],
        "sal_psu": df[c_S],
        "basin_atomic": df[c_basin],
        "lon": df[c_lon] if c_lon else np.nan,
        "lat": df[c_lat] if c_lat else np.nan,
    })

    # Much faster than full pd.to_datetime for giant files
    out["year"] = pd.to_numeric(
        out["time_raw"].astype(str).str.slice(0, 4),
        errors="coerce"
    ).astype("float32")

    out = out.drop(columns=["time_raw"])

    out["depth_m"] = pd.to_numeric(out["depth_m"], errors="coerce").astype("float32")
    out["temp_c"]  = pd.to_numeric(out["temp_c"], errors="coerce").astype("float32")
    out["sal_psu"] = pd.to_numeric(out["sal_psu"], errors="coerce").astype("float32")
    out["lon"]     = pd.to_numeric(out["lon"], errors="coerce").astype("float32")
    out["lat"]     = pd.to_numeric(out["lat"], errors="coerce").astype("float32")

    out["basin_atomic"] = (
        out["basin_atomic"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    return out


def build_atomic_to_composed_map(comp: dict) -> dict[str, str]:
    m: dict[str, str] = {}
    for comp_id, info in comp.items():
        for b in info.get("members", []):
            m[str(b).strip().upper()] = str(comp_id).strip().lower()
    return m


def subsample(df: pd.DataFrame, nmax: int | None, seed: int) -> pd.DataFrame:
    if (nmax is None) or (len(df) <= nmax):
        return df
    return df.sample(n=nmax, random_state=seed)


def plot_depth_markers(ax, df: pd.DataFrame, cmap_name: str, norm: Normalize) -> None:
    if df.empty:
        return

    cmap = plt.get_cmap(cmap_name)

    for z in DEPTH_MARKERS:
        sub = df[
            (df["depth_m"] >= z - DEPTH_MARKER_HALF_WINDOW) &
            (df["depth_m"] <= z + DEPTH_MARKER_HALF_WINDOW)
        ]
        if len(sub) < DEPTH_MARKER_MIN_N:
            continue

        s_med = float(sub["sal_psu"].median())
        t_med = float(sub["temp_c"].median())
        z_med = float(sub["depth_m"].median())
        face = cmap(norm(z_med))

        ax.scatter(
            s_med, t_med,
            s=65,
            marker="o",
            facecolor=face,
            edgecolor="black",
            linewidth=0.6,
            zorder=6,
        )


def _compute_sigma0_grid(SP_grid, t_grid, lon=None, lat=None):
    try:
        import gsw  # type: ignore
    except Exception as e:
        raise ImportError("Need 'gsw' package for PDA isolines. Install with: pip install gsw") from e

    p0 = np.zeros_like(SP_grid)

    if lon is not None and lat is not None and np.isfinite(lon) and np.isfinite(lat):
        SA = gsw.SA_from_SP(SP_grid, p0, lon, lat)
        CT = gsw.CT_from_t(SA, t_grid, p0)
        sig0 = gsw.sigma0(SA, CT)
        return sig0

    return gsw.sigma0(SP_grid, t_grid)


def add_pda_isolines(ax, d_for_loc: pd.DataFrame | None, levels,
                     *, linewidth=0.8, alpha=0.45, do_labels=True):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    S = np.linspace(x0, x1, 140)
    T = np.linspace(y0, y1, 140)
    SPg, Tg = np.meshgrid(S, T)

    lon = lat = None
    if d_for_loc is not None and ("lon" in d_for_loc.columns) and ("lat" in d_for_loc.columns):
        lon_med = pd.to_numeric(d_for_loc["lon"], errors="coerce").median()
        lat_med = pd.to_numeric(d_for_loc["lat"], errors="coerce").median()
        if np.isfinite(lon_med) and np.isfinite(lat_med):
            lon, lat = float(lon_med), float(lat_med)

    sig0 = _compute_sigma0_grid(SPg, Tg, lon=lon, lat=lat)

    cs = ax.contour(
        SPg, Tg, sig0,
        levels=levels,
        colors="0.35",
        linewidths=linewidth,
        alpha=alpha,
        zorder=1,
    )
    if do_labels:
        ax.clabel(cs, inline=True, fontsize=9, fmt="%.1f")


def load_or_build_cached_dataframe() -> pd.DataFrame:
    os.makedirs(OUTDIR, exist_ok=True)

    if os.path.exists(CACHE_PARQUET):
        print(f"Reading cached parquet: {CACHE_PARQUET}")
        return pd.read_parquet(CACHE_PARQUET)

    print("Cache not found. Building dataframe from source CSVs...")

    frames: list[pd.DataFrame] = []

    atomic_to_comp = build_atomic_to_composed_map(COMPOSED_BASINS)

    for fp in INPUT_CSVS:
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)

        print(f"Reading source: {fp}")

        colmap = detect_columns_in_file(fp)
        usecols, dtype_map = build_usecols_and_dtypes(colmap)

        try:
            df0 = pd.read_csv(
                fp,
                usecols=usecols,
                dtype=dtype_map,
                low_memory=False,
                engine="pyarrow"
            )
        except Exception:
            df0 = pd.read_csv(
                fp,
                usecols=usecols,
                dtype=dtype_map,
                low_memory=False
            )

        dfi = standardize_one(df0, colmap)

        if BASIN_COMPOSED_COL is not None:
            if BASIN_COMPOSED_COL not in dfi.columns:
                raise ValueError(f"BASIN_COMPOSED_COL='{BASIN_COMPOSED_COL}' not found in dataframe.")
            dfi["basin_composed"] = dfi[BASIN_COMPOSED_COL].astype(str).str.strip().str.lower()
        else:
            dfi["basin_composed"] = dfi["basin_atomic"].map(atomic_to_comp)

        # Early filtering = faster + less RAM
        dfi = dfi.dropna(subset=["year", "depth_m", "temp_c", "sal_psu", "basin_composed"])
        dfi = dfi[dfi["basin_composed"].isin(REGIONS)]
        dfi = dfi[dfi["year"] >= P20_START]
        dfi = dfi[dfi["year"] <= P21_END]
        dfi = dfi[dfi["depth_m"] >= DEPTH_MIN]

        if DEPTH_MAX is not None:
            dfi = dfi[dfi["depth_m"] <= DEPTH_MAX]

        dfi = dfi[(dfi["temp_c"] > -2) & (dfi["temp_c"] < 40)]
        dfi = dfi[(dfi["sal_psu"] >= 30) & (dfi["sal_psu"] <= 42)]

        # Keep only what is actually needed later
        dfi = dfi[[
            "year", "depth_m", "temp_c", "sal_psu",
            "basin_composed", "lon", "lat"
        ]].copy()

        frames.append(dfi)

        del df0, dfi

    df = pd.concat(frames, ignore_index=True)

    print(f"Saving parquet cache: {CACHE_PARQUET}")
    df.to_parquet(CACHE_PARQUET, index=False)

    return df

# ==========================================================
# Main
# ==========================================================

def main() -> None:
    if not INPUT_CSVS:
        raise SystemExit("No input CSVs found in multi_source_seasonal_climatology.SOURCES.")

    df = load_or_build_cached_dataframe()

    # Split by period
    df20 = df[(df["year"] >= P20_START) & (df["year"] <= P20_END)].copy()
    df21 = df[(df["year"] >= P21_START) & (df["year"] <= P21_END)].copy()

    # Shared depth normalization
    zmin = float(DEPTH_MIN)
    zmax = float(DEPTH_MAX) if DEPTH_MAX is not None else float(np.nanpercentile(df["depth_m"].values, 99))
    norm = Normalize(vmin=zmin, vmax=zmax)

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=False)
    axes = axes.ravel()

    rng = np.random.default_rng(RANDOM_SEED)

    for i, region in enumerate(REGIONS):
        ax = axes[i]

        d20 = df20[df20["basin_composed"] == region]
        d21 = df21[df21["basin_composed"] == region]

        d20 = subsample(d20, MAX_POINTS_PER_REGION_PERIOD, seed=int(rng.integers(0, 1_000_000)))
        d21 = subsample(d21, MAX_POINTS_PER_REGION_PERIOD, seed=int(rng.integers(0, 1_000_000)))

        lims = AXIS_LIMITS.get(region)
        if lims:
            ax.set_xlim(*lims["S"])
            ax.set_ylim(*lims["T"])

        if PLOT_PDA_ISOLINES:
            d_loc = pd.concat([d20, d21], ignore_index=True) if (len(d20) or len(d21)) else None
            add_pda_isolines(
                ax, d_loc, PDA_LEVELS,
                linewidth=PDA_LINEWIDTH,
                alpha=PDA_ALPHA,
                do_labels=PDA_LABELS,
            )

        if len(d20):
            ax.scatter(
                d20["sal_psu"], d20["temp_c"],
                c=d20["depth_m"], cmap="Blues", norm=norm,
                s=SCATTER_S, alpha=SCATTER_ALPHA, marker=".", linewidths=0,
                rasterized=True,
            )

        if len(d21):
            ax.scatter(
                d21["sal_psu"], d21["temp_c"],
                c=d21["depth_m"], cmap="Reds", norm=norm,
                s=SCATTER_S, alpha=SCATTER_ALPHA, marker=".", linewidths=0,
                rasterized=True,
            )

        if PLOT_DEPTH_MARKERS:
            plot_depth_markers(ax, d20, cmap_name="Blues", norm=norm)
            plot_depth_markers(ax, d21, cmap_name="Reds", norm=norm)

        n20 = len(d20)
        n21 = len(d21)

        ax.text(
            0.5, 1.065,
            f"{region.upper()} — {COMPOSED_BASINS[region]['name']}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold"
        )

        ax.text(
            0.5, 1.015,
            f"(20c n={n20:,}, 21c n={n21:,})",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="normal"
        )

        ax.set_xlabel("Salinity", labelpad=6)
        ax.set_ylabel("Temperature (°C)", labelpad=6)

        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=4,
            width=1,
            pad=3
        )

        ax.grid(True, alpha=0.15, linewidth=0.6)

    fig.suptitle(
        "T–S properties below seasonal thermocline (z ≥ 100 m)",
        fontsize=13,
        fontweight="bold",
        y=0.965
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.08,
        top=0.89,
        wspace=0.22,
        hspace=0.30
    )

    fig.canvas.draw()

    sm20 = ScalarMappable(norm=norm, cmap="Blues")
    sm20.set_array([])

    sm21 = ScalarMappable(norm=norm, cmap="Reds")
    sm21.set_array([])

    pos00 = axes[0].get_position()
    pos01 = axes[1].get_position()
    pos10 = axes[2].get_position()
    pos11 = axes[3].get_position()

    right_edge = max(pos01.x1, pos11.x1)
    cbar_pad = 0.015
    cbar_width = 0.015
    cbar_x = right_edge + cbar_pad

    top_row_y0 = min(pos00.y0, pos01.y0)
    top_row_y1 = max(pos00.y1, pos01.y1)
    top_row_h = top_row_y1 - top_row_y0

    cax1 = fig.add_axes([cbar_x, top_row_y0, cbar_width, top_row_h])
    cb1 = fig.colorbar(sm20, cax=cax1)
    cb1.ax.tick_params(labelsize=10, width=1)
    cb1.set_label("Depth (m) — 20c", fontsize=11, weight="bold")

    bot_row_y0 = min(pos10.y0, pos11.y0)
    bot_row_y1 = max(pos10.y1, pos11.y1)
    bot_row_h = bot_row_y1 - bot_row_y0

    cax2 = fig.add_axes([cbar_x, bot_row_y0, cbar_width, bot_row_h])
    cb2 = fig.colorbar(sm21, cax=cax2)
    cb2.ax.tick_params(labelsize=10, width=1)
    cb2.set_label("Depth (m) — 21c", fontsize=11, weight="bold")

    fig.savefig(OUTPNG, dpi=DPI)
    plt.close(fig)

    print(f"\nSaved: {OUTPNG}")
    print(f"Cache used/built: {CACHE_PARQUET}")

    print("\nCounts per region:")
    for r in REGIONS:
        n20 = (df20["basin_composed"] == r).sum()
        n21 = (df21["basin_composed"] == r).sum()
        print(f"  {r}: 20c={n20:,}  21c={n21:,}")


if __name__ == "__main__":
    main()