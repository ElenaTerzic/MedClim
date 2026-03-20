#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v6 FAST: Exclude profiles/casts whose station position is influenced by:

1) MAINLAND proximity only (islands ignored for this rule)
2) Special geometric masks:
   - tuned river-mouth circular masks
   - tuned box masks (straits, deltas, estuarine areas)
3) Inland points (inside mainland polygon)

Final exclusion logic
---------------------
exclude_final = inside_mainland_polygon | exclude_near_mainland | special_mask_hit

Notes
-----
- Distances are computed in EPSG:3035 (meters).
- Filtering is done at profile/cast level.
- Output CSV preserves original columns/structure; only rows belonging to
  excluded casts are removed.
- This script does NOT use low-salinity plume detection yet.

Speed improvements
------------------
- Special-circle masking is vectorized with NumPy.
- Box masking is vectorized.
- Special-mask geometries are prepared once and reused.
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import box
from shapely.ops import unary_union


# ==========================================================
# USER SETTINGS
# ==========================================================

SUFFIX = "_mainland_special_excluded"

SAVE_PROFILE_DIAGNOSTICS = True
SAVE_QC_MAP = True

METRIC_CRS = "EPSG:3035"
MEDITERRANEAN_BBOX = (-12.0, 18.0, 42.5, 52.0)  # lon_min, lat_min, lon_max, lat_max

# Mainland-only mask construction
MAINLAND_MIN_AREA_KM2 = 50000.0
MAINLAND_MIN_AREA_M2 = MAINLAND_MIN_AREA_KM2 * 1e6

# Default/variable mainland proximity rule
DEFAULT_DIST_KM = 2.0

BASIN_DIST_KM = {
    "DJ1": 1.0,
}

# ==========================================================
# TUNED SPECIAL MASKS
# copied from test_estuary_locations.py
# ==========================================================

SPECIAL_MASK_RULES = [
    {"type": "circle", "name": "Rhone",            "lon": 4.80,     "lat": 43.35,    "radius_km": 20.0},
    {"type": "circle", "name": "Ebro",             "lon": 0.83,     "lat": 40.72,    "radius_km": 20.0},

    {"type": "box",    "name": "Po_delta_box",     "lon_min": 12.20, "lon_max": 12.70, "lat_min": 44.70, "lat_max": 45.15},

    {"type": "circle", "name": "Nile_Rosetta",     "lon": 30.36702, "lat": 31.46485, "radius_km": 20.0},
    {"type": "circle", "name": "Nile_Damietta",    "lon": 31.84372, "lat": 31.52178, "radius_km": 20.0},

    {"type": "circle", "name": "Drin_Bojana",      "lon": 19.30,    "lat": 41.85,    "radius_km": 10.0},

    {"type": "box",    "name": "Neretva_1",        "lon_min": 17.30, "lon_max": 17.52, "lat_min": 42.92, "lat_max": 43.08},
    {"type": "box",    "name": "Neretva_2",        "lon_min": 17.45, "lon_max": 17.62, "lat_min": 42.88, "lat_max": 42.97},
    {"type": "box",    "name": "Neretva_3",        "lon_min": 17.62, "lon_max": 17.78, "lat_min": 42.80, "lat_max": 42.88},

    {"type": "circle", "name": "Tiber",            "lon": 12.23,    "lat": 41.74,    "radius_km": 10.0},
    {"type": "circle", "name": "Adige_r8",         "lon": 12.33,    "lat": 45.16,    "radius_km": 10.0},
    {"type": "circle", "name": "Seyhan",           "lon": 34.91,    "lat": 36.73,    "radius_km": 12.0},
    {"type": "circle", "name": "Ceyhan",           "lon": 35.55,    "lat": 36.56,    "radius_km": 12.0},

    {"type": "box",    "name": "Krka_box",         "lon_min": 15.73, "lon_max": 15.90, "lat_min": 43.65, "lat_max": 43.76},
    {"type": "box",    "name": "Zrmanja_box",      "lon_min": 15.30, "lon_max": 15.65, "lat_min": 44.10, "lat_max": 44.40},
    {"type": "box",    "name": "Dardanelles_box",  "lon_min": 26.10, "lon_max": 27.15, "lat_min": 39.98, "lat_max": 40.55},
]

# QC plotting
QC_MAP_MAX_POINTS = 50000
QC_PAD_DEG = 0.25
QC_MARKER_KEPT = 6
QC_MARKER_EXCL = 8
QC_MARKER_SUSPECT = 18
QC_LABEL_MASKS = True

# Natural Earth cache
CACHE_DIR = Path.home() / ".cache" / "medclim_mainland_mask"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
NE_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip"


# ==========================================================
# SOURCE DEFINITIONS
# ==========================================================

GOOD_QF_SDN = {1, 2}
GOOD_QF_ARGO = {1, 2, 5, 8}


@dataclass
class SourceSpec:
    name: str
    path: Path
    time_col: str
    basin_col: str
    depth_col: str
    temp_col: str
    sal_col: str
    cast_id_cols: List[str]
    lon_col: str
    lat_col: str
    time_format: Optional[str] = None
    add_date_to_cast_id: bool = False
    temp_qc_col: Optional[str] = None
    sal_qc_col: Optional[str] = None
    depth_qc_col: Optional[str] = None
    good_qf: Optional[set] = None
    basin_remap: Optional[dict] = None
    depth_kind: str = "depth_m"


SOURCES: List[SourceSpec] = [
    SourceSpec(
        name="SDN",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins.csv"),
        time_col="yyyy-mm-ddThh:mm:ss.sss",
        basin_col="basin",
        depth_col="Depth [m]",
        lon_col="Longitude [degrees_east]",
        lat_col="Latitude [degrees_north]",
        temp_col="ITS-90 water temperature [degrees C]",
        sal_col="Water body salinity [per mille]",
        cast_id_cols=[
            "LOCAL_CDI_ID||LOCAL_CDI_ID:METAVAR:INDEXED_TEXT",
            "Cruise",
            "Station:METAVAR:INDEXED_TEXT",
        ],
        add_date_to_cast_id=False,
        temp_qc_col="QV:SEADATANET.1",
        sal_qc_col="QV:SEADATANET.2",
        good_qf=GOOD_QF_SDN,
        basin_remap=None,
        depth_kind="depth_m",
    ),
    SourceSpec(
        name="ARGO",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Argo/CSV/Argo_with_basins.csv"),
        time_col="TIME",
        time_format="%Y-%m-%dT%H:%M:%S",
        basin_col="basin",
        depth_col="PRES",
        temp_col="TEMP",
        sal_col="PSAL",
        cast_id_cols=["PLATFORM_NUMBER", "CYCLE_NUMBER"],
        add_date_to_cast_id=False,
        temp_qc_col="TEMP_QC",
        sal_qc_col="PSAL_QC",
        depth_qc_col="PRES_QC",
        good_qf=GOOD_QF_ARGO,
        basin_remap=None,
        depth_kind="pres_dbar",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
    ),
    SourceSpec(
        name="CTD_HR_2019_2023_MFSD",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/CTD_HR_2019_2023_MFSD_long_fixed.csv"),
        time_col="TIME",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        cast_id_cols=["STATION"],
        add_date_to_cast_id=True,
        basin_remap=None,
        depth_kind="depth_m",
    ),
    SourceSpec(
        name="RB_Northern_Adriatic_Transect_1979_2018",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/RB_Northern_Adriatic_Transect_1979_2018_long_fixed.csv"),
        time_col="TIME",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        cast_id_cols=["STATION"],
        add_date_to_cast_id=True,
        basin_remap=None,
        depth_kind="depth_m",
    ),
    SourceSpec(
        name="Palagruza_Transect_1952_2018",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/Palagruza_Transect_1952_2018_long_fixed.csv"),
        time_col="TIME",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        cast_id_cols=["STATION"],
        add_date_to_cast_id=True,
        basin_remap=None,
        depth_kind="depth_m",
    ),
    SourceSpec(
        name="IOR_Adriatic_database_1987_2019",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/IOR_Adriatic_database_1987_2019_long_fixed.csv"),
        time_col="TIME",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        basin_col="BASIN_3ADR_FIXED",
        depth_col="DEPTH_M",
        temp_col="TEMP_C",
        sal_col="SAL_PSU",
        cast_id_cols=["STATION", "CAST_IDX"],
        add_date_to_cast_id=True,
        basin_remap=None,
        depth_kind="depth_m",
    ),
]


# ==========================================================
# HELPERS
# ==========================================================

def ensure_natural_earth_land() -> Path:
    zip_path = CACHE_DIR / Path(NE_URL).name
    if not zip_path.exists():
        print(f"[INFO] Downloading Natural Earth land polygons:\n       {NE_URL}")
        urllib.request.urlretrieve(NE_URL, zip_path)
    return zip_path


def output_path_with_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix + path.suffix)


def choose_first_existing_column(df: pd.DataFrame, expr: str) -> str:
    candidates = [x.strip() for x in expr.split("||")]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}")


def build_cast_uid(df: pd.DataFrame, spec: SourceSpec) -> pd.Series:
    parts = []

    for expr in spec.cast_id_cols:
        col = choose_first_existing_column(df, expr)
        vals = df[col].astype(str).fillna("NA")
        parts.append(vals)

    if spec.add_date_to_cast_id:
        t = pd.to_datetime(df[spec.time_col], errors="coerce")
        date_str = t.dt.strftime("%Y%m%d").fillna("NA_DATE")
        parts.append(date_str)

    uid = parts[0].copy()
    for p in parts[1:]:
        uid = uid + "||" + p

    return spec.name + "||" + uid


def fix_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    try:
        from shapely import make_valid
        gdf["geometry"] = gdf.geometry.apply(make_valid)
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf


def build_mainland_polygons_and_coastline() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    zip_path = ensure_natural_earth_land()

    land = gpd.read_file(f"zip://{zip_path}")
    if land.crs is None:
        land = land.set_crs("EPSG:4326")
    else:
        land = land.to_crs("EPSG:4326")

    med_box = box(*MEDITERRANEAN_BBOX)

    land = land[land.intersects(med_box)].copy()
    land["geometry"] = land.geometry.intersection(med_box)
    land = fix_geometries(land)

    land = land.explode(index_parts=False, ignore_index=True)
    land = land[land.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    land = fix_geometries(land)

    land_metric = land.to_crs(METRIC_CRS).copy()
    land_metric["area_m2"] = land_metric.geometry.area

    mainland = land_metric.loc[land_metric["area_m2"] >= MAINLAND_MIN_AREA_M2].copy()
    if mainland.empty:
        raise RuntimeError("No mainland polygons survived the area threshold.")

    mainland = fix_geometries(mainland)

    mainland_union = unary_union(mainland.geometry.values)
    mainland_coastline = mainland_union.boundary

    mainland_gdf = gpd.GeoDataFrame(
        {"name": ["mainland"], "geometry": [mainland_union]},
        crs=METRIC_CRS,
    )
    coastline_gdf = gpd.GeoDataFrame(
        {"name": ["mainland_coastline"], "geometry": [mainland_coastline]},
        crs=METRIC_CRS,
    )
    return mainland_gdf, coastline_gdf


def maybe_downsample_for_plot(gdf: gpd.GeoDataFrame, max_points: int) -> gpd.GeoDataFrame:
    if len(gdf) <= max_points:
        return gdf
    step = max(1, math.ceil(len(gdf) / max_points))
    return gdf.iloc[::step].copy()


def assign_basin_exclusion_km(
    df: pd.DataFrame,
    basin_col: str,
) -> pd.Series:
    dist_km = pd.Series(DEFAULT_DIST_KM, index=df.index, dtype=float)

    if basin_col in df.columns and BASIN_DIST_KM:
        basin_series = df[basin_col].astype(str)
        for basin_code, value in BASIN_DIST_KM.items():
            dist_km.loc[basin_series == str(basin_code)] = float(value)

    return dist_km


# ==========================================================
# FAST SPECIAL MASK PREP
# ==========================================================

def build_special_mask_geometries_and_arrays():
    circle_rows = []
    box_rows = []

    circle_rule_names = []
    circle_rule_x = []
    circle_rule_y = []
    circle_rule_r_m = []

    box_rule_names = []
    box_rule_lon_min = []
    box_rule_lon_max = []
    box_rule_lat_min = []
    box_rule_lat_max = []

    for rule in SPECIAL_MASK_RULES:
        rtype = str(rule["type"]).lower()

        if rtype == "circle":
            center_ll = gpd.GeoSeries(
                gpd.points_from_xy([rule["lon"]], [rule["lat"]]),
                crs="EPSG:4326",
            )
            center_metric = center_ll.to_crs(METRIC_CRS).iloc[0]
            radius_m = float(rule["radius_km"]) * 1000.0

            circle_rows.append(
                {
                    "name": rule["name"],
                    "mask_type": "circle",
                    "radius_km": float(rule["radius_km"]),
                    "geometry": center_metric.buffer(radius_m),
                }
            )

            circle_rule_names.append(rule["name"])
            circle_rule_x.append(center_metric.x)
            circle_rule_y.append(center_metric.y)
            circle_rule_r_m.append(radius_m)

        elif rtype == "box":
            geom_ll = box(rule["lon_min"], rule["lat_min"], rule["lon_max"], rule["lat_max"])
            box_rows.append(
                {
                    "name": rule["name"],
                    "mask_type": "box",
                    "radius_km": np.nan,
                    "geometry": geom_ll,
                }
            )

            box_rule_names.append(rule["name"])
            box_rule_lon_min.append(float(rule["lon_min"]))
            box_rule_lon_max.append(float(rule["lon_max"]))
            box_rule_lat_min.append(float(rule["lat_min"]))
            box_rule_lat_max.append(float(rule["lat_max"]))

        else:
            raise ValueError(f"Unknown SPECIAL_MASK_RULES type: {rtype}")

    if circle_rows:
        circle_gdf = gpd.GeoDataFrame(circle_rows, crs=METRIC_CRS)
    else:
        circle_gdf = gpd.GeoDataFrame(
            columns=["name", "mask_type", "radius_km", "geometry"],
            geometry="geometry",
            crs=METRIC_CRS,
        )

    if box_rows:
        box_gdf = gpd.GeoDataFrame(box_rows, crs="EPSG:4326").to_crs(METRIC_CRS)
    else:
        box_gdf = gpd.GeoDataFrame(
            columns=["name", "mask_type", "radius_km", "geometry"],
            geometry="geometry",
            crs=METRIC_CRS,
        )

    prepared = {
        "circle_names": np.array(circle_rule_names, dtype=object),
        "circle_x": np.array(circle_rule_x, dtype=float),
        "circle_y": np.array(circle_rule_y, dtype=float),
        "circle_r_m": np.array(circle_rule_r_m, dtype=float),

        "box_names": np.array(box_rule_names, dtype=object),
        "box_lon_min": np.array(box_rule_lon_min, dtype=float),
        "box_lon_max": np.array(box_rule_lon_max, dtype=float),
        "box_lat_min": np.array(box_rule_lat_min, dtype=float),
        "box_lat_max": np.array(box_rule_lat_max, dtype=float),
    }

    return circle_gdf, box_gdf, prepared


# ==========================================================
# FAST SPECIAL MASK APPLICATION
# ==========================================================

def apply_circle_masks_fast(
    gprof: gpd.GeoDataFrame,
    prepared_masks: dict,
) -> pd.DataFrame:
    out = pd.DataFrame(index=gprof.index)
    out["circle_mask_hit"] = False
    out["circle_mask_name"] = ""
    out["circle_mask_dist_km"] = np.nan

    circle_x = prepared_masks["circle_x"]
    circle_y = prepared_masks["circle_y"]
    circle_r_m = prepared_masks["circle_r_m"]
    circle_names = prepared_masks["circle_names"]

    if circle_x.size == 0 or len(gprof) == 0:
        return out

    px = gprof.geometry.x.to_numpy(dtype=float)
    py = gprof.geometry.y.to_numpy(dtype=float)

    dx = px[:, None] - circle_x[None, :]
    dy = py[:, None] - circle_y[None, :]
    dist_m = np.hypot(dx, dy)

    hit_matrix = dist_m <= circle_r_m[None, :]
    any_hit = hit_matrix.any(axis=1)
    first_idx = np.argmax(hit_matrix, axis=1)

    hit_rows = np.where(any_hit)[0]
    if hit_rows.size > 0:
        chosen = first_idx[hit_rows]
        out.iloc[hit_rows, out.columns.get_loc("circle_mask_hit")] = True
        out.iloc[hit_rows, out.columns.get_loc("circle_mask_name")] = circle_names[chosen]
        out.iloc[hit_rows, out.columns.get_loc("circle_mask_dist_km")] = dist_m[hit_rows, chosen] / 1000.0

    return out


def apply_box_masks_fast(
    gprof: gpd.GeoDataFrame,
    lon_col: str,
    lat_col: str,
    prepared_masks: dict,
) -> pd.DataFrame:
    out = pd.DataFrame(index=gprof.index)
    out["box_mask_hit"] = False
    out["box_mask_name"] = ""

    box_names = prepared_masks["box_names"]
    if box_names.size == 0 or len(gprof) == 0:
        return out

    lon = gprof[lon_col].to_numpy(dtype=float)
    lat = gprof[lat_col].to_numpy(dtype=float)

    lon_min = prepared_masks["box_lon_min"]
    lon_max = prepared_masks["box_lon_max"]
    lat_min = prepared_masks["box_lat_min"]
    lat_max = prepared_masks["box_lat_max"]

    hit_matrix = (
        (lon[:, None] >= lon_min[None, :]) &
        (lon[:, None] <= lon_max[None, :]) &
        (lat[:, None] >= lat_min[None, :]) &
        (lat[:, None] <= lat_max[None, :])
    )

    any_hit = hit_matrix.any(axis=1)
    first_idx = np.argmax(hit_matrix, axis=1)

    hit_rows = np.where(any_hit)[0]
    if hit_rows.size > 0:
        chosen = first_idx[hit_rows]
        out.iloc[hit_rows, out.columns.get_loc("box_mask_hit")] = True
        out.iloc[hit_rows, out.columns.get_loc("box_mask_name")] = box_names[chosen]

    return out


def combine_special_masks_fast(
    gprof: gpd.GeoDataFrame,
    lon_col: str,
    lat_col: str,
    prepared_masks: dict,
) -> gpd.GeoDataFrame:
    circle_hits = apply_circle_masks_fast(gprof, prepared_masks)
    box_hits = apply_box_masks_fast(gprof, lon_col=lon_col, lat_col=lat_col, prepared_masks=prepared_masks)

    gprof = gprof.join(circle_hits)
    gprof = gprof.join(box_hits)

    gprof["special_mask_hit"] = gprof["circle_mask_hit"] | gprof["box_mask_hit"]

    gprof["special_mask_name"] = ""
    gprof.loc[gprof["circle_mask_hit"], "special_mask_name"] = gprof.loc[gprof["circle_mask_hit"], "circle_mask_name"]
    gprof.loc[
        (~gprof["circle_mask_hit"]) & (gprof["box_mask_hit"]),
        "special_mask_name"
    ] = gprof.loc[(~gprof["circle_mask_hit"]) & (gprof["box_mask_hit"]), "box_mask_name"]

    gprof["special_mask_type"] = ""
    gprof.loc[gprof["circle_mask_hit"], "special_mask_type"] = "circle"
    gprof.loc[
        (~gprof["circle_mask_hit"]) & (gprof["box_mask_hit"]),
        "special_mask_type"
    ] = "box"

    return gprof


def basin_summary_table(prof: pd.DataFrame, basin_col: str) -> str:
    rows = []
    for basin, sub in prof.groupby(basin_col, dropna=False):
        n_prof = len(sub)
        n_excl = int(sub["exclude_final"].sum())
        n_keep = n_prof - n_excl
        pct = (100.0 * n_excl / n_prof) if n_prof > 0 else np.nan
        rows.append((str(basin), n_prof, n_excl, n_keep, pct))

    rows = sorted(rows, key=lambda x: x[0])

    out = []
    out.append("    Per-basin profile summary:")
    out.append("    basin                     n_profiles   excluded   kept   pct_excluded")
    out.append("    ----------------------------------------------------------------------")
    for basin, n_prof, n_excl, n_keep, pct in rows:
        out.append(f"    {basin:<24} {n_prof:>10d} {n_excl:>10d} {n_keep:>7d} {pct:>13.2f}")
    return "\n".join(out)


def save_qc_map(
    spec: SourceSpec,
    mainland_gdf: gpd.GeoDataFrame,
    circle_mask_gdf: gpd.GeoDataFrame,
    box_mask_gdf: gpd.GeoDataFrame,
    gprof_metric: gpd.GeoDataFrame,
    out_png: Path,
) -> None:
    if gprof_metric.empty:
        return

    def _clean_for_plot(gdf: gpd.GeoDataFrame, to_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        if gdf is None or len(gdf) == 0:
            return gpd.GeoDataFrame(geometry=[], crs=to_crs)

        gg = gdf.copy()
        try:
            from shapely import make_valid
            gg["geometry"] = gg.geometry.apply(make_valid)
        except Exception:
            gg["geometry"] = gg.geometry.buffer(0)

        gg = gg[gg.geometry.notna()].copy()
        gg = gg[~gg.geometry.is_empty].copy()

        if len(gg) == 0:
            return gpd.GeoDataFrame(geometry=[], crs=to_crs)

        gg = gg.to_crs(to_crs)

        bounds = gg.geometry.bounds
        finite_mask = np.isfinite(bounds).all(axis=1)
        gg = gg.loc[finite_mask].copy()

        if len(gg) == 0:
            return gpd.GeoDataFrame(geometry=[], crs=to_crs)

        return gg

    gprof_plot = _clean_for_plot(gprof_metric)
    mainland_plot = _clean_for_plot(mainland_gdf)
    circle_plot = _clean_for_plot(circle_mask_gdf)
    box_plot = _clean_for_plot(box_mask_gdf)

    max_dist_m = float(gprof_metric["local_dist_m"].max()) if "local_dist_m" in gprof_metric.columns else DEFAULT_DIST_KM * 1000.0
    coast_buffer_metric = mainland_gdf.copy()
    coast_buffer_metric["geometry"] = coast_buffer_metric.geometry.buffer(max_dist_m)
    coast_buffer_plot = _clean_for_plot(coast_buffer_metric)

    if gprof_plot.empty:
        return

    gprof_plot = maybe_downsample_for_plot(gprof_plot, QC_MAP_MAX_POINTS)

    kept = gprof_plot.loc[~gprof_plot["exclude_final"]].copy()
    excl_mainland_only = gprof_plot.loc[
        gprof_plot["exclude_near_mainland"] & (~gprof_plot["special_mask_hit"]) & (~gprof_plot["inside_mainland_polygon"])
    ].copy()
    excl_special = gprof_plot.loc[gprof_plot["special_mask_hit"]].copy()

    suspect = gprof_plot.iloc[0:0].copy()
    if "inside_mainland_polygon" in gprof_plot.columns:
        suspect = gprof_plot.loc[gprof_plot["inside_mainland_polygon"]].copy()

    xmin = gprof_plot.geometry.x.min()
    xmax = gprof_plot.geometry.x.max()
    ymin = gprof_plot.geometry.y.min()
    ymax = gprof_plot.geometry.y.max()

    fig, ax = plt.subplots(figsize=(8.8, 8.8))
    ax.set_aspect("auto")

    if not coast_buffer_plot.empty:
        coast_buffer_plot.plot(ax=ax, alpha=0.15, edgecolor="none", aspect="auto")
    if not circle_plot.empty:
        circle_plot.plot(ax=ax, alpha=0.18, edgecolor="black", linewidth=0.5, aspect="auto")
    if not box_plot.empty:
        box_plot.boundary.plot(ax=ax, linewidth=1.0, linestyle="--", aspect="auto")
    if not mainland_plot.empty:
        mainland_plot.boundary.plot(ax=ax, linewidth=0.7, color="black", aspect="auto")

    if not excl_mainland_only.empty:
        excl_mainland_only.plot(ax=ax, markersize=QC_MARKER_EXCL, label="excluded: mainland", aspect="auto")

    if not excl_special.empty:
        excl_special.plot(ax=ax, markersize=QC_MARKER_EXCL + 2, marker="s", label="excluded: special mask", aspect="auto")

    if not kept.empty:
        kept.plot(ax=ax, markersize=QC_MARKER_KEPT, label="kept", aspect="auto")

    if not suspect.empty:
        suspect.plot(ax=ax, markersize=QC_MARKER_SUSPECT, marker="x", label="inside mainland polygon", aspect="auto")

    if QC_LABEL_MASKS:
        for _, row in circle_plot.iterrows():
            try:
                rp = row.geometry.representative_point()
                if np.isfinite(rp.x) and np.isfinite(rp.y):
                    ax.text(rp.x, rp.y, row["name"], fontsize=7, ha="center", va="center")
            except Exception:
                pass

        for _, row in box_plot.iterrows():
            try:
                rp = row.geometry.representative_point()
                if np.isfinite(rp.x) and np.isfinite(rp.y):
                    ax.text(rp.x, rp.y, row["name"], fontsize=7, ha="center", va="center")
            except Exception:
                pass

    ax.set_title(f"{spec.name} – mainland + special masks QC")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(xmin - QC_PAD_DEG, xmax + QC_PAD_DEG)
    ax.set_ylim(ymin - QC_PAD_DEG, ymax + QC_PAD_DEG)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ==========================================================
# PROCESSING
# ==========================================================

def process_source(
    spec: SourceSpec,
    mainland_gdf: gpd.GeoDataFrame,
    coastline_gdf: gpd.GeoDataFrame,
    circle_mask_gdf: gpd.GeoDataFrame,
    box_mask_gdf: gpd.GeoDataFrame,
    prepared_masks: dict,
) -> None:
    print("=" * 100)
    print(f"[SOURCE] {spec.name}")
    print(f"[INPUT ] {spec.path}")

    if not spec.path.exists():
        print("[WARN  ] File not found, skipping.")
        return

    df = pd.read_csv(spec.path)

    required = [spec.lon_col, spec.lat_col, spec.time_col, spec.basin_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{spec.name}: missing required columns: {missing}")

    df["_cast_uid"] = build_cast_uid(df, spec)

    prof = (
        df[["_cast_uid", spec.lon_col, spec.lat_col, spec.basin_col]]
        .drop_duplicates(subset="_cast_uid")
        .copy()
    )

    prof[spec.lon_col] = pd.to_numeric(prof[spec.lon_col], errors="coerce")
    prof[spec.lat_col] = pd.to_numeric(prof[spec.lat_col], errors="coerce")
    prof["_valid_xy"] = prof[spec.lon_col].notna() & prof[spec.lat_col].notna()

    n_profiles_total = len(prof)
    n_profiles_invalid_xy = int((~prof["_valid_xy"]).sum())

    prof_valid = prof.loc[prof["_valid_xy"]].copy()

    gprof = gpd.GeoDataFrame(
        prof_valid,
        geometry=gpd.points_from_xy(prof_valid[spec.lon_col], prof_valid[spec.lat_col]),
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)

    mainland_coast = coastline_gdf.geometry.iloc[0]
    mainland_union = mainland_gdf.geometry.iloc[0]

    gprof["dist_to_mainland_m"] = gprof.geometry.distance(mainland_coast)

    gprof["local_dist_km"] = assign_basin_exclusion_km(
        gprof,
        basin_col=spec.basin_col,
    )
    gprof["local_dist_m"] = 1000.0 * gprof["local_dist_km"]
    gprof["exclude_near_mainland"] = gprof["dist_to_mainland_m"] <= gprof["local_dist_m"]

    gprof = combine_special_masks_fast(
        gprof,
        lon_col=spec.lon_col,
        lat_col=spec.lat_col,
        prepared_masks=prepared_masks,
    )

    gprof["inside_mainland_polygon"] = gprof.geometry.within(mainland_union)

    gprof["exclude_final"] = (
        gprof["inside_mainland_polygon"] |
        gprof["exclude_near_mainland"] |
        gprof["special_mask_hit"]
    )

    merge_cols = [
        "_cast_uid",
        "dist_to_mainland_m",
        "local_dist_km",
        "local_dist_m",
        "exclude_near_mainland",
        "circle_mask_hit",
        "circle_mask_name",
        "circle_mask_dist_km",
        "box_mask_hit",
        "box_mask_name",
        "special_mask_hit",
        "special_mask_name",
        "special_mask_type",
        "exclude_final",
        "inside_mainland_polygon",
    ]

    prof = prof.merge(
        gprof[merge_cols],
        on="_cast_uid",
        how="left",
    )

    prof["exclude_near_mainland"] = prof["exclude_near_mainland"].fillna(False)
    prof["circle_mask_hit"] = prof["circle_mask_hit"].fillna(False)
    prof["box_mask_hit"] = prof["box_mask_hit"].fillna(False)
    prof["special_mask_hit"] = prof["special_mask_hit"].fillna(False)
    prof["exclude_final"] = prof["exclude_final"].fillna(False)
    prof["inside_mainland_polygon"] = prof["inside_mainland_polygon"].fillna(False)

    df = df.merge(
        prof[merge_cols],
        on="_cast_uid",
        how="left",
    )

    n_profiles_excluded = int(prof["exclude_final"].sum())
    n_profiles_kept = n_profiles_total - n_profiles_excluded
    n_rows_total = len(df)
    n_rows_excluded = int(df["exclude_final"].sum())
    n_rows_kept = n_rows_total - n_rows_excluded
    n_inland = int(prof["inside_mainland_polygon"].sum())
    n_mainland = int(prof["exclude_near_mainland"].sum())
    n_special = int(prof["special_mask_hit"].sum())

    print(f"[INFO  ] Total profiles                          : {n_profiles_total}")
    print(f"[INFO  ] Profiles with invalid coords            : {n_profiles_invalid_xy}")
    print(f"[INFO  ] Excluded profiles (final)               : {n_profiles_excluded}")
    print(f"[INFO  ] Kept profiles                           : {n_profiles_kept}")
    print(f"[INFO  ] Total rows                              : {n_rows_total}")
    print(f"[INFO  ] Excluded rows (final)                   : {n_rows_excluded}")
    print(f"[INFO  ] Kept rows                               : {n_rows_kept}")
    print(f"[INFO  ] Profiles excluded by mainland rule      : {n_mainland}")
    print(f"[INFO  ] Profiles excluded by special masks      : {n_special}")
    print(f"[INFO  ] Profiles inside mainland polygon        : {n_inland}")
    print()

    print("[INFO  ] Local exclusion distances used (km):")
    print(gprof["local_dist_km"].value_counts().sort_index().to_string())
    print()

    if n_special > 0:
        print("[INFO  ] Special mask hits by name:")
        print(gprof.loc[gprof["special_mask_hit"], "special_mask_name"].value_counts().to_string())
        print()

    print(basin_summary_table(prof, spec.basin_col))
    print()

    if n_inland > 0:
        preview = gprof.loc[
            gprof["inside_mainland_polygon"],
            ["_cast_uid", spec.lon_col, spec.lat_col, spec.basin_col, "dist_to_mainland_m", "local_dist_km"]
        ].head(20).copy()
        if not preview.empty:
            preview["dist_to_mainland_km"] = preview["dist_to_mainland_m"] / 1000.0
            print("[WARN  ] First suspicious profiles inside mainland polygon:")
            print(preview[["_cast_uid", spec.lon_col, spec.lat_col, spec.basin_col, "dist_to_mainland_km", "local_dist_km"]].to_string(index=False))
            print()

    out_csv = output_path_with_suffix(spec.path, SUFFIX)
    drop_cols = [
        "_cast_uid",
        "dist_to_mainland_m",
        "local_dist_km",
        "local_dist_m",
        "exclude_near_mainland",
        "circle_mask_hit",
        "circle_mask_name",
        "circle_mask_dist_km",
        "box_mask_hit",
        "box_mask_name",
        "special_mask_hit",
        "special_mask_name",
        "special_mask_type",
        "exclude_final",
        "inside_mainland_polygon",
    ]
    original_cols = [c for c in df.columns if c not in drop_cols]

    df_out = df.loc[~df["exclude_final"], original_cols].copy()
    df_out.to_csv(out_csv, index=False)
    print(f"[OUTPUT] Filtered CSV: {out_csv}")

    if SAVE_PROFILE_DIAGNOSTICS:
        diag_csv = output_path_with_suffix(spec.path, SUFFIX + "_profile_diagnostics")
        prof_diag = prof.copy()
        prof_diag["dist_to_mainland_km"] = prof_diag["dist_to_mainland_m"] / 1000.0
        prof_diag.to_csv(diag_csv, index=False)
        print(f"[OUTPUT] Profile diagnostics: {diag_csv}")

    if SAVE_QC_MAP and not gprof.empty:
        qc_png = output_path_with_suffix(spec.path, SUFFIX + "_qc_map").with_suffix(".png")
        save_qc_map(spec, mainland_gdf, circle_mask_gdf, box_mask_gdf, gprof, qc_png)
        print(f"[OUTPUT] QC map: {qc_png}")

    print()


def main() -> None:
    print("[INFO] Building mainland-only coastline mask...")
    mainland_gdf, coastline_gdf = build_mainland_polygons_and_coastline()
    circle_mask_gdf, box_mask_gdf, prepared_masks = build_special_mask_geometries_and_arrays()

    print("[DEBUG ] Circle mask count before plotting:", len(circle_mask_gdf))
    print("[DEBUG ] Box mask count before plotting   :", len(box_mask_gdf))
    print("[DEBUG ] Circle mask bounds:")
    if len(circle_mask_gdf) > 0:
        print(circle_mask_gdf.to_crs("EPSG:4326").total_bounds)
    else:
        print("[]")
    print("[DEBUG ] Box mask bounds:")
    if len(box_mask_gdf) > 0:
        print(box_mask_gdf.to_crs("EPSG:4326").total_bounds)
    else:
        print("[]")
    print()

    print("[INFO] Mainland mask ready.")
    print(f"[INFO] Mainland min area threshold = {MAINLAND_MIN_AREA_KM2:.0f} km²")
    print(f"[INFO] Default mainland exclusion distance = {DEFAULT_DIST_KM:.1f} km")
    print()

    print("[INFO] Basin overrides:")
    if BASIN_DIST_KM:
        for k, v in BASIN_DIST_KM.items():
            print(f"         {k}: {v:.1f} km")
    else:
        print("         none")
    print()

    print("[INFO] Special masks:")
    for rule in SPECIAL_MASK_RULES:
        if rule["type"] == "circle":
            print(f"         {rule['name']}: circle ({rule['lon']:.3f}, {rule['lat']:.3f}), r={rule['radius_km']:.1f} km")
        else:
            print(
                f"         {rule['name']}: "
                f"box lon[{rule['lon_min']}, {rule['lon_max']}], "
                f"lat[{rule['lat_min']}, {rule['lat_max']}]"
            )
    print()

    for spec in SOURCES:
        process_source(
            spec,
            mainland_gdf,
            coastline_gdf,
            circle_mask_gdf,
            box_mask_gdf,
            prepared_masks,
        )

    print("=" * 100)
    print("[DONE] All sources processed.")


if __name__ == "__main__":
    main()