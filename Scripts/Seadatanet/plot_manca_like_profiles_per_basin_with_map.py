#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- MAP LIBS (coastlines/countries) ---
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------------
# INPUTS (BOTH PERIODS)
# -----------------------------
INPUTS = [
    (
        "20c_1950_1999",
        Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/climatology_sdn_20c_1950_1999_after_1sigma.csv"),
    ),
    (
        "21c_2000_2019",
        Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/climatology_sdn_21c_2000_2019_after_1sigma.csv"),
    ),
]

OUTBASE = Path("/Users/elena/Projects/IRB/MedClim/Plots/Seadatanet") / "Seasonal_Profiles_Per_Basin"
OUTBASE.mkdir(parents=True, exist_ok=True)

# -----------------------------
# BASINS (your working approach)
# -----------------------------
import Basins

BASIN_LIST = [
    Basins.manca.df2, Basins.manca.df3, Basins.manca.df4, Basins.manca.ds2, Basins.manca.df1,
    Basins.manca.ds1, Basins.manca.ds3, Basins.manca.ds4, Basins.manca.dt1, Basins.manca.dt2,
    Basins.manca.dt3, Basins.manca.di1, Basins.manca.di3, Basins.manca.dj1, Basins.manca.dj2,
    Basins.manca.dj3, Basins.manca.dj6, Basins.manca.dj4, Basins.manca.dj7, Basins.manca.dj8,
    Basins.manca.dj5, Basins.manca.dh1, Basins.manca.dh2, Basins.manca.dh3, Basins.manca.dl1,
    Basins.manca.dl2, Basins.manca.dl3, Basins.manca.dl4
]
BASIN_BY_ABBREV = {b.abbrev.upper(): b for b in BASIN_LIST}

# -----------------------------
# CONFIG
# -----------------------------
INSET_ZMAX = 200.0

MIN_N_FOR_ZMAX = 10
MAX_PLOT_Z = 2000.0
MIN_PLOT_Z = 200.0

SEASON_MAP = {
    "JFM": "Winter",
    "AMJ": "Spring",
    "JAS": "Summer",
    "OND": "Autumn",
    "WINTER": "Winter",
    "SPRING": "Spring",
    "SUMMER": "Summer",
    "AUTUMN": "Autumn",
}
SEASONS_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

COLORS = {
    "Winter": "tab:blue",
    "Spring": "tab:green",
    "Summer": "tab:red",
    "Autumn": "tab:orange",
}

INSET_CFG = {
    "T": dict(width="38%", height="38%", loc="lower left", bbox_to_anchor=(0.18, 0.12, 1, 1)),
    "S": dict(width="38%", height="38%", loc="lower left", bbox_to_anchor=(0.16, 0.12, 1, 1)),
}

# Map view (Mediterranean)
MAP_EXTENT = (-6, 36.5, 30, 46)  # lon_min, lon_max, lat_min, lat_max

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

# -----------------------------
# HELPERS
# -----------------------------
def _std_shade(ax, x, y, sd, color, alpha=0.15):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(sd)
    if m.any():
        ax.fill_betweenx(y[m], (x - sd)[m], (x + sd)[m], color=color, alpha=alpha, linewidth=0)

def _add_inset(parent_ax, var_key: str):
    cfg = INSET_CFG[var_key]
    return inset_axes(
        parent_ax,
        width=cfg["width"],
        height=cfg["height"],
        loc=cfg["loc"],
        bbox_to_anchor=cfg["bbox_to_anchor"],
        bbox_transform=parent_ax.transAxes,
        borderpad=0.8,
    )

def compute_basin_zmax(df_basin: pd.DataFrame) -> float:
    g = df_basin.copy()
    g["ok"] = (g["n_T"] >= MIN_N_FOR_ZMAX) | (g["n_S"] >= MIN_N_FOR_ZMAX)
    depths_ok = g.loc[g["ok"], "z_m"].to_numpy(float)
    if depths_ok.size == 0:
        return MIN_PLOT_Z
    zmax = float(np.nanmax(depths_ok))
    return max(MIN_PLOT_Z, min(MAX_PLOT_Z, zmax))

def _basin_bbox_from_obj(b):
    """
    Try to get bbox from basin object.
    Falls back to scanning grid points if needed.
    Expected: b has either xmin/xmax/ymin/ymax or similar.
    """
    for cand in [("xmin","xmax","ymin","ymax"), ("x0","x1","y0","y1")]:
        if all(hasattr(b, k) for k in cand):
            xmin, xmax, ymin, ymax = [float(getattr(b, k)) for k in cand]
            return xmin, xmax, ymin, ymax
    # fallback: try areinside on a grid
    lon = np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], 400)
    lat = np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], 250)
    LON, LAT = np.meshgrid(lon, lat)
    pts = np.column_stack([LON.ravel(), LAT.ravel(), np.zeros(LON.size)])
    try:
        inside = np.asarray(b.areinside(pts)).astype(bool)
        if inside.any():
            LONi = pts[inside, 0]
            LATi = pts[inside, 1]
            return float(LONi.min()), float(LONi.max()), float(LATi.min()), float(LATi.max())
    except Exception:
        pass
    return None

def _draw_basin_outline(ax, b, edgecolor="#7e57c2", lw=2.0):
    """
    Prefer a coastline-following outline if basin object provides a polygon.
    If not available, fallback to bbox outline.
    """
    # 1) If basin has an explicit polygon attribute, use it (common names)
    for attr in ("poly", "polygon", "outline", "boundary"):
        if hasattr(b, attr):
            poly = getattr(b, attr)
            # poly could be list of (lon,lat) or Nx2 array
            try:
                arr = np.asarray(poly)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    ax.plot(arr[:, 0], arr[:, 1], transform=ccrs.PlateCarree(),
                            color=edgecolor, linewidth=lw)
                    return True
            except Exception:
                pass

    # 2) Fallback: bbox
    bb = _basin_bbox_from_obj(b)
    if bb is None:
        return False
    xmin, xmax, ymin, ymax = bb
    xs = [xmin, xmax, xmax, xmin, xmin]
    ys = [ymin, ymin, ymax, ymax, ymin]
    ax.plot(xs, ys, transform=ccrs.PlateCarree(), color=edgecolor, linewidth=lw)
    return True

def plot_one_basin(df_basin: pd.DataFrame, basin_code: str, tag: str, outdir: Path):
    zmax = compute_basin_zmax(df_basin)

    # --- 3 panels: T | S | Map ---
    fig = plt.figure(figsize=(16.0, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.35], wspace=0.25)

    axT = fig.add_subplot(gs[0, 0])
    axS = fig.add_subplot(gs[0, 1], sharey=axT)
    axM = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())

    fig.suptitle(f"{basin_code}: Seasonal climatology ({tag}) (mean ± std on main panels)")

    # Main axes formatting
    for ax in (axT, axS):
        ax.set_ylim(zmax, 0)
        ax.grid(True, alpha=0.35)
        ax.set_ylabel("Depth [m]")

    axT.set_title("Temperature [°C]")
    axS.set_title("Salinity [-]")
    axT.set_xlabel("Temperature [°C]")
    axS.set_xlabel("Salinity [-]")

    # Insets
    iaT = _add_inset(axT, "T")
    iaS = _add_inset(axS, "S")
    for iax in (iaT, iaS):
        iax.set_ylim(INSET_ZMAX, 0)
        iax.grid(True, alpha=0.35)
        iax.tick_params(labelsize=10)
    iaT.set_title(f"0–{int(INSET_ZMAX)} m", fontsize=11)
    iaS.set_title(f"0–{int(INSET_ZMAX)} m", fontsize=11)

    any_plotted = False

    for season in SEASONS_ORDER:
        g = df_basin[df_basin["season_name"] == season].sort_values("z_m")
        if g.empty:
            continue

        z = g["z_m"].to_numpy(float)

        # Temperature
        Tm = g["T_mean"].to_numpy(float)
        Ts = g["T_std"].to_numpy(float)
        mT = np.isfinite(z) & np.isfinite(Tm) & (z >= 0) & (z <= zmax)
        if mT.any():
            axT.plot(Tm[mT], z[mT], color=COLORS[season], lw=2.0, label=season)
            _std_shade(axT, Tm[mT], z[mT], Ts[mT], COLORS[season])
            any_plotted = True

        # Salinity
        Sm = g["S_mean"].to_numpy(float)
        Ss = g["S_std"].to_numpy(float)
        mS = np.isfinite(z) & np.isfinite(Sm) & (z >= 0) & (z <= zmax)
        if mS.any():
            axS.plot(Sm[mS], z[mS], color=COLORS[season], lw=2.0, label=season)
            _std_shade(axS, Sm[mS], z[mS], Ss[mS], COLORS[season])
            any_plotted = True

        # Insets: mean only, 0–200
        mi = np.isfinite(z) & (z >= 0) & (z <= INSET_ZMAX)
        if mi.any():
            if np.isfinite(Tm[mi]).any():
                iaT.plot(Tm[mi], z[mi], color=COLORS[season], lw=1.6)
            if np.isfinite(Sm[mi]).any():
                iaS.plot(Sm[mi], z[mi], color=COLORS[season], lw=1.6)

    if any_plotted:
        axS.legend(loc="lower right", frameon=True)

    # --- MAP PANEL ---
    axM.set_title("Basin location", fontsize=18)
    axM.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    # Thin, “nice” coastlines + countries
    axM.add_feature(cfeature.LAND, facecolor="0.96", edgecolor="none", zorder=0)
    axM.coastlines(resolution="50m", linewidth=0.7, color="0.2", zorder=2)
    axM.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, edgecolor="0.5", zorder=2)

    gl = axM.gridlines(draw_labels=True, linewidth=0.4, color="0.7", alpha=0.6, linestyle="-")
    gl.top_labels = False
    gl.right_labels = False

    # Basin outline
    bobj = BASIN_BY_ABBREV.get(basin_code.upper())
    if bobj is None:
        axM.text(0.5, 0.5, f"No basin object for {basin_code}\n(check Basins.manca list)",
                 ha="center", va="center", transform=axM.transAxes, fontsize=13)
    else:
        ok = _draw_basin_outline(axM, bobj, lw=2.0)
        if not ok:
            axM.text(0.5, 0.5, f"Could not draw outline for {basin_code}",
                     ha="center", va="center", transform=axM.transAxes, fontsize=13)

        axM.text(0.02, 0.06, basin_code, transform=axM.transAxes,
                 fontsize=18, weight="bold", color="black")

    # Layout + save
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    outpng = outdir / f"{basin_code}_seasonal_profiles_TS_{tag}.png"
    fig.savefig(outpng, dpi=200)
    plt.close(fig)
    print("Wrote:", outpng)

def load_and_clean(infile: Path) -> pd.DataFrame:
    df = pd.read_csv(infile)

    need = {"basin", "season", "z_m", "n_T", "T_mean", "T_std", "n_S", "S_mean", "S_std"}
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise SystemExit(f"Missing columns in {infile}:\n{missing}")

    df["season"] = df["season"].astype(str).str.strip()
    df["season_name"] = df["season"].str.upper().map(SEASON_MAP)

    df["z_m"] = pd.to_numeric(df["z_m"], errors="coerce")
    for c in ["n_T", "n_S"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ["T_mean", "T_std", "S_mean", "S_std"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["basin", "season_name", "z_m"])
    df["basin"] = df["basin"].astype(str).str.strip()
    return df

def main():
    for tag, infile in INPUTS:
        if not infile.exists():
            print(f"WARNING: missing input: {infile}")
            continue

        outdir = OUTBASE / tag
        outdir.mkdir(parents=True, exist_ok=True)

        print("\n====================================")
        print("Plotting:", tag)
        print("Input:   ", infile)
        print("Outdir:  ", outdir)
        print("====================================")

        df = load_and_clean(infile)
        basins = sorted(df["basin"].astype(str).unique().tolist())
        print(f"Basins found in CSV: {len(basins)}")

        # Optional: show which basins are missing basin-objects
        missing = [b for b in basins if b.upper() not in BASIN_BY_ABBREV]
        if missing:
            print("WARNING: missing basin objects for:", missing)

        for basin_code in basins:
            plot_one_basin(df[df["basin"] == basin_code].copy(), basin_code, tag, outdir)

if __name__ == "__main__":
    main()
