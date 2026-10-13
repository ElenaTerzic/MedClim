#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reads : Data/{Seadatanet,Argo/CSV,Adriatic/Processed}/*_mainland_special_excluded_profile_diagnostics.csv (before the Argo-copy removal)
# Writes: Plots/CoastalExclusionQC/*.png, Zooms_actual_masks_split_causes/*.png, BasinZooms/{Atomic,Composed}/*.png

"""
QC plotting for exclusion masks and excluded points.

This version shows the ACTUAL filtering mask geometries used in the
modified remove_river_discharges.py script, with exclusion causes split
visually into separate point categories:

1) mainland / default coastal exclusion buffer (2 km visualized globally)
2) tuned circle masks
3) tuned box masks
4) kept points
5) points excluded specifically by 2 km mainland rule
6) points excluded specifically by DJ1 1 km mainland rule
7) points excluded by special masks / mixed special overlaps
8) inland points
9) zoom plots for key regions
10) atomic basin maps
11) composed basin maps

Outputs
-------
1) 2x3 panel: all datasets separately
2) Combined Mediterranean map
3) Zoom plots per target region
4) One map per atomic basin
5) One map per composed basin

Important
---------
- Red circles = actual circle masks used in filtering
- Blue dashed boxes = actual box masks used in filtering
- Light blue shaded zone = default mainland coastal buffer (2 km)
- Beige = mainland
- River lines = contextual only
- Green points = kept
- Cyan circles = excluded by 2 km mainland rule only
- Orange triangles = excluded by DJ1 1 km mainland rule only
- Black squares = excluded by special masks (or overlaps involving special masks)
- Magenta x = inland points
- Diagnostics CSVs are read from the filtering script output

Notes
-----
- The filtering script uses 2 km by default, but 1 km for DJ1.
  Here the coastal buffer layer is drawn as a global default 2 km buffer
  for visualization simplicity.
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from pyproj import Geod
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Scripts.paths import DATA, PLOTS


# ==========================================================
# USER SETTINGS
# ==========================================================

OUTPUT_DIR = PLOTS / "CoastalExclusionQC"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PANEL = OUTPUT_DIR / "QC_actual_masks_2x3_split_causes.png"
OUT_COMBINED = OUTPUT_DIR / "QC_actual_masks_combined_med_split_causes.png"

SAVE_ZOOM_PLOTS = True
ZOOM_OUTPUT_DIR = OUTPUT_DIR / "Zooms_actual_masks_split_causes"
ZOOM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_BASIN_MAPS = True
BASIN_OUTPUT_DIR = OUTPUT_DIR / "BasinZooms"
ATOMIC_OUTPUT_DIR = BASIN_OUTPUT_DIR / "Atomic"
COMPOSED_OUTPUT_DIR = BASIN_OUTPUT_DIR / "Composed"
ATOMIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPOSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHOW_RIVER_LINES = True
SHOW_CIRCLE_CENTERS = True
SHOW_CIRCLE_LABELS_IN_COMBINED = True
SHOW_CIRCLE_LABELS_IN_ZOOMS = True
SHOW_CIRCLE_LABELS_IN_BASINS = True
SHOW_BOX_LABELS_IN_COMBINED = True
SHOW_BOX_LABELS_IN_ZOOMS = True
SHOW_BOX_LABELS_IN_BASINS = True

SHOW_MAINLAND_2KM_POINTS = True
SHOW_DJ1_1KM_POINTS = True
SHOW_SPECIAL_MASK_POINTS = True
SHOW_INLAND_POINTS = True

MAINLAND_2KM_COLOR = "cyan"
MAINLAND_2KM_MARKER = "o"
MAINLAND_2KM_SIZE = 16

DJ1_1KM_COLOR = "orange"
DJ1_1KM_MARKER = "^"
DJ1_1KM_SIZE = 20

SPECIAL_MASK_COLOR = "black"
SPECIAL_MASK_MARKER = "s"
SPECIAL_MASK_SIZE = 18

INLAND_COLOR = "magenta"
INLAND_MARKER = "x"
INLAND_SIZE = 24

KEPT_COLOR = "forestgreen"
KEPT_SIZE = 5

CIRCLE_CENTER_MARKER_SIZE = 48
CIRCLE_LABEL_SIZE = 8
CIRCLE_LABEL_CONNECTOR_WIDTH = 0.8

PANEL_FIGSIZE = (18, 10)
COMBINED_FIGSIZE = (12, 9)
ZOOM_FIGSIZE = (8, 7)
BASIN_FIGSIZE = (8, 7)

QC_PAD_DEG = 0.20
BASIN_PAD_DEG = 0.20
MAX_POINTS_PER_DATASET = 30000
MAX_POINTS_PER_BASIN = 50000

ZOOM_POINT_SIZE_KEPT = 12
ZOOM_MAINLAND_2KM_SIZE = 28
ZOOM_DJ1_1KM_SIZE = 32
ZOOM_SPECIAL_MASK_SIZE = 30
ZOOM_INLAND_SIZE = 36

BASIN_POINT_SIZE_KEPT = 12
BASIN_MAINLAND_2KM_SIZE = 28
BASIN_DJ1_1KM_SIZE = 32
BASIN_SPECIAL_MASK_SIZE = 30
BASIN_INLAND_SIZE = 36

LAND_FACE = "#eadfcf"
SEA_FACE = "#eef5fb"
COAST_COLOR = "black"

METRIC_CRS = "EPSG:3035"
MEDITERRANEAN_BBOX = (-12.0, 18.0, 42.5, 52.0)
MAINLAND_MIN_AREA_KM2 = 50000.0
MAINLAND_MIN_AREA_M2 = MAINLAND_MIN_AREA_KM2 * 1e6

DEFAULT_DIST_KM = 2.0

CACHE_DIR = Path.home() / ".cache" / "medclim_mainland_mask"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NE_LAND_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip"
NE_RIVERS_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_rivers_lake_centerlines.zip"


# ==========================================================
# ACTUAL FILTERING MASKS
# Must match the modified filtering script
# ==========================================================

SPECIAL_MASK_RULES = [
    {"type": "circle", "name": "Rhone",            "lon": 4.80,     "lat": 43.35,    "radius_km": 20.0},
    {"type": "circle", "name": "Ebro",             "lon": 0.83,     "lat": 40.72,    "radius_km": 20.0},

    {"type": "box",    "name": "Po_delta_box",     "label": "Po_delta_box",
     "lon_min": 12.20, "lon_max": 12.70, "lat_min": 44.70, "lat_max": 45.15},

    {"type": "circle", "name": "Nile_Rosetta",     "lon": 30.36702, "lat": 31.46485, "radius_km": 20.0},
    {"type": "circle", "name": "Nile_Damietta",    "lon": 31.84372, "lat": 31.52178, "radius_km": 20.0},

    {"type": "circle", "name": "Drin_Bojana",      "lon": 19.30,    "lat": 41.85,    "radius_km": 10.0},

    {"type": "box",    "name": "Neretva_1",        "label": "Neretva_1",
     "lon_min": 17.30, "lon_max": 17.52, "lat_min": 42.92, "lat_max": 43.08},
    {"type": "box",    "name": "Neretva_2",        "label": "Neretva_2",
     "lon_min": 17.45, "lon_max": 17.62, "lat_min": 42.88, "lat_max": 42.97},
    {"type": "box",    "name": "Neretva_3",        "label": "Neretva_3",
     "lon_min": 17.62, "lon_max": 17.78, "lat_min": 42.80, "lat_max": 42.88},

    {"type": "circle", "name": "Tiber",            "lon": 12.23,    "lat": 41.74,    "radius_km": 10.0},
    {"type": "circle", "name": "Adige_r8",         "lon": 12.33,    "lat": 45.16,    "radius_km": 10.0},
    {"type": "circle", "name": "Seyhan",           "lon": 34.91,    "lat": 36.73,    "radius_km": 12.0},
    {"type": "circle", "name": "Ceyhan",           "lon": 35.55,    "lat": 36.56,    "radius_km": 12.0},

    {"type": "box",    "name": "Krka_box",         "label": "Krka_box",
     "lon_min": 15.73, "lon_max": 15.90, "lat_min": 43.65, "lat_max": 43.76},
    {"type": "box",    "name": "Zrmanja_box",      "label": "Zrmanja_box",
     "lon_min": 15.30, "lon_max": 15.65, "lat_min": 44.10, "lat_max": 44.40},
    {"type": "box",    "name": "Dardanelles_box",  "label": "Dardanelles_box",
     "lon_min": 26.10, "lon_max": 27.15, "lat_min": 39.98, "lat_max": 40.55},
]


# ==========================================================
# COMPOSED BASINS
# ==========================================================

COMPOSED_BASINS: Dict[str, Dict[str, object]] = {
    "alb":  {"name": "Alboran",            "members": ["DS1"]},
    "west": {"name": "Western Med",        "members": ["DS2", "DS3", "DS4", "DI1"]},
    "nwm":  {"name": "North-Western Med",  "members": ["DF1", "DF2", "DF3", "DF4"]},
    "tyr":  {"name": "Tyrrhenian",         "members": ["DT1", "DT2", "DT3"]},
    "sic":  {"name": "Sicily Strait",      "members": ["DI3"]},

    "nadr": {"name": "Northern Adriatic",  "members": ["DJ1"]},
    "madr": {"name": "Mid Adriatic",       "members": ["DJ2"]},
    "sadr": {"name": "Southern Adriatic",  "members": ["DJ3"]},

    "ion":  {"name": "Ionian",             "members": ["DJ4", "DJ5", "DJ6", "DJ7", "DJ8"]},

    "lev":  {"name": "Levantine",          "members": ["DL1", "DL2", "DL3", "DL4", "DH3"]},

    "naeg": {"name": "Northern Aegean",    "members": ["DH1"]},
    "saeg": {"name": "Southern Aegean",    "members": ["DH2"]},
}


# ==========================================================
# ZOOM REGIONS
# ==========================================================

ZOOM_REGIONS = [
    {"name": "Rhone",        "lon_min": 4.0,   "lon_max": 5.6,   "lat_min": 42.9,  "lat_max": 43.9},
    {"name": "Ebro",         "lon_min": 0.1,   "lon_max": 1.5,   "lat_min": 40.2,  "lat_max": 41.2},
    {"name": "Po",           "lon_min": 12.0,  "lon_max": 12.9,  "lat_min": 44.6,  "lat_max": 45.2},
    {"name": "Nile",         "lon_min": 29.5,  "lon_max": 32.5,  "lat_min": 30.8,  "lat_max": 32.0},
    {"name": "Drin_Bojana",  "lon_min": 18.9,  "lon_max": 19.7,  "lat_min": 41.6,  "lat_max": 42.2},
    {"name": "Neretva",      "lon_min": 17.2,  "lon_max": 17.9,  "lat_min": 42.75, "lat_max": 43.12},
    {"name": "Tiber",        "lon_min": 11.9,  "lon_max": 12.6,  "lat_min": 41.5,  "lat_max": 42.0},
    {"name": "Adige",        "lon_min": 12.0,  "lon_max": 12.7,  "lat_min": 44.9,  "lat_max": 45.3},
    {"name": "Seyhan",       "lon_min": 34.5,  "lon_max": 35.4,  "lat_min": 36.4,  "lat_max": 37.0},
    {"name": "Ceyhan",       "lon_min": 35.1,  "lon_max": 35.9,  "lat_min": 36.3,  "lat_max": 36.8},
    {"name": "Krka",         "lon_min": 15.65, "lon_max": 15.98, "lat_min": 43.60, "lat_max": 43.82},
    {"name": "Zrmanja",      "lon_min": 15.20, "lon_max": 15.72, "lat_min": 44.05, "lat_max": 44.42},
    {"name": "Dardanelles",  "lon_min": 25.9,  "lon_max": 27.3,  "lat_min": 39.9,  "lat_max": 40.7},
]

CIRCLE_LABEL_OFFSETS = {
    "Rhone":         (0.25,  0.12),
    "Ebro":          (0.22, -0.10),
    "Nile_Rosetta":  (-0.55, 0.12),
    "Nile_Damietta": (0.18,  0.12),
    "Drin_Bojana":   (0.20, -0.12),
    "Tiber":         (0.18, -0.10),
    "Adige_r8":      (0.18,  0.10),
    "Seyhan":        (0.18,  0.10),
    "Ceyhan":        (0.18, -0.10),
}

BOX_LABEL_OFFSETS = {
    "Po_delta_box":    (0.00, 0.00),
    "Neretva_1":       (0.00, 0.00),
    "Neretva_2":       (0.00, 0.00),
    "Neretva_3":       (0.00, 0.00),
    "Krka_box":        (0.00, 0.00),
    "Zrmanja_box":     (0.00, 0.00),
    "Dardanelles_box": (0.10, 0.12),
}


# ==========================================================
# DATASETS
# ==========================================================

@dataclass
class DiagSpec:
    name: str
    path: Path
    lon_col: str
    lat_col: str
    basin_col: str


DATASETS: List[DiagSpec] = [
    DiagSpec(
        name="SDN",
        path=DATA / "Seadatanet" / "Seadatanet.profiles_only_with_basins_mainland_special_excluded_profile_diagnostics.csv",
        lon_col="Longitude [degrees_east]",
        lat_col="Latitude [degrees_north]",
        basin_col="basin",
    ),
    DiagSpec(
        name="ARGO",
        path=DATA / "Argo" / "CSV" / "Argo_with_basins_mainland_special_excluded_profile_diagnostics.csv",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        basin_col="basin",
    ),
    DiagSpec(
        name="CTD_HR_2019_2023_MFSD",
        path=DATA / "Adriatic" / "Processed" / "CTD_HR_2019_2023_MFSD_long_fixed_mainland_special_excluded_profile_diagnostics.csv",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        basin_col="BASIN_3ADR_FIXED",
    ),
    DiagSpec(
        name="RB_Northern_Adriatic_Transect_1979_2018",
        path=DATA / "Adriatic" / "Processed" / "RB_Northern_Adriatic_Transect_1979_2018_long_fixed_mainland_special_excluded_profile_diagnostics.csv",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        basin_col="BASIN_3ADR_FIXED",
    ),
    DiagSpec(
        name="Palagruza_Transect_1952_2018",
        path=DATA / "Adriatic" / "Processed" / "Palagruza_Transect_1952_2018_long_fixed_mainland_special_excluded_profile_diagnostics.csv",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        basin_col="BASIN_3ADR_FIXED",
    ),
    DiagSpec(
        name="IOR_Adriatic_database_1987_2019",
        path=DATA / "Adriatic" / "Processed" / "IOR_Adriatic_database_1987_2019_long_fixed_mainland_special_excluded_profile_diagnostics.csv",
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
        basin_col="BASIN_3ADR_FIXED",
    ),
]


# ==========================================================
# FILE HELPERS
# ==========================================================

def ensure_natural_earth_land() -> Path:
    zip_path = CACHE_DIR / Path(NE_LAND_URL).name
    if not zip_path.exists():
        urllib.request.urlretrieve(NE_LAND_URL, zip_path)
    return zip_path


def ensure_natural_earth_rivers() -> Path:
    zip_path = CACHE_DIR / Path(NE_RIVERS_URL).name
    if not zip_path.exists():
        urllib.request.urlretrieve(NE_RIVERS_URL, zip_path)
    return zip_path


# ==========================================================
# GEOMETRY HELPERS
# ==========================================================

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


def clean_for_plot(gdf: gpd.GeoDataFrame, to_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    if gdf is None:
        return gpd.GeoDataFrame(geometry=[], crs=to_crs)

    base_cols = [c for c in gdf.columns if c != "geometry"]

    if len(gdf) == 0:
        data = {c: [] for c in base_cols}
        data["geometry"] = []
        return gpd.GeoDataFrame(data, geometry="geometry", crs=to_crs)

    gg = gdf.copy()

    try:
        from shapely import make_valid
        gg["geometry"] = gg.geometry.apply(make_valid)
    except Exception:
        gg["geometry"] = gg.geometry.buffer(0)

    gg = gg[gg.geometry.notna()].copy()
    gg = gg[~gg.geometry.is_empty].copy()

    if len(gg) == 0:
        data = {c: [] for c in base_cols}
        data["geometry"] = []
        return gpd.GeoDataFrame(data, geometry="geometry", crs=to_crs)

    gg = gg.to_crs(to_crs)

    bounds = gg.geometry.bounds
    finite_mask = np.isfinite(bounds).all(axis=1)
    gg = gg.loc[finite_mask].copy()

    if len(gg) == 0:
        data = {c: [] for c in base_cols}
        data["geometry"] = []
        return gpd.GeoDataFrame(data, geometry="geometry", crs=to_crs)

    return gg


def maybe_downsample_for_plot(gdf: gpd.GeoDataFrame, max_points: int) -> gpd.GeoDataFrame:
    if len(gdf) <= max_points:
        return gdf
    step = max(1, math.ceil(len(gdf) / max_points))
    return gdf.iloc[::step].copy()


def build_mainland_polygons() -> gpd.GeoDataFrame:
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
    mainland = fix_geometries(mainland)

    mainland_union = unary_union(mainland.geometry.values)
    return gpd.GeoDataFrame({"name": ["mainland"], "geometry": [mainland_union]}, crs=METRIC_CRS)


def build_river_lines() -> gpd.GeoDataFrame:
    zip_path = ensure_natural_earth_rivers()
    rivers = gpd.read_file(f"zip://{zip_path}")

    if rivers.crs is None:
        rivers = rivers.set_crs("EPSG:4326")
    else:
        rivers = rivers.to_crs("EPSG:4326")

    med_box = box(*MEDITERRANEAN_BBOX)
    rivers = rivers[rivers.intersects(med_box)].copy()
    rivers["geometry"] = rivers.geometry.intersection(med_box)
    rivers = fix_geometries(rivers)
    return clean_for_plot(rivers, to_crs="EPSG:4326")


def make_geodesic_circle_polygon(lon: float, lat: float, radius_km: float, n_points: int = 180) -> Polygon:
    geod = Geod(ellps="WGS84")
    azimuths = np.linspace(0, 360, n_points, endpoint=False)
    coords = []
    for az in azimuths:
        lon2, lat2, _ = geod.fwd(lon, lat, az, radius_km * 1000.0)
        coords.append((lon2, lat2))
    coords.append(coords[0])
    return Polygon(coords)


def build_circle_mask_geometries() -> gpd.GeoDataFrame:
    rows = []
    for rule in SPECIAL_MASK_RULES:
        if rule["type"] != "circle":
            continue
        rows.append(
            {
                "name": rule["name"],
                "geometry": make_geodesic_circle_polygon(rule["lon"], rule["lat"], rule["radius_km"], n_points=180),
            }
        )
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return clean_for_plot(gdf, to_crs="EPSG:4326")


def build_box_mask_geometries() -> gpd.GeoDataFrame:
    rows = []
    for rule in SPECIAL_MASK_RULES:
        if rule["type"] != "box":
            continue
        rows.append(
            {
                "name": rule["name"],
                "label": rule.get("label", rule["name"]),
                "geometry": box(rule["lon_min"], rule["lat_min"], rule["lon_max"], rule["lat_max"]),
            }
        )
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return clean_for_plot(gdf, to_crs="EPSG:4326")


def build_circle_center_points() -> gpd.GeoDataFrame:
    rows = []
    for rule in SPECIAL_MASK_RULES:
        if rule["type"] != "circle":
            continue
        rows.append(
            {
                "name": rule["name"],
                "lon": rule["lon"],
                "lat": rule["lat"],
                "radius_km": rule["radius_km"],
                "geometry": Point(rule["lon"], rule["lat"]),
            }
        )
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return clean_for_plot(gdf, to_crs="EPSG:4326")


# ==========================================================
# DATA LOADING
# ==========================================================

def load_diag_as_gdf(spec: DiagSpec) -> gpd.GeoDataFrame | None:
    if not spec.path.exists():
        print(f"[WARN] Missing diagnostics CSV for {spec.name}: {spec.path}")
        return None

    df = pd.read_csv(spec.path)

    if spec.lon_col not in df.columns or spec.lat_col not in df.columns:
        print(f"[WARN] Missing lon/lat columns for {spec.name}")
        return None

    df[spec.lon_col] = pd.to_numeric(df[spec.lon_col], errors="coerce")
    df[spec.lat_col] = pd.to_numeric(df[spec.lat_col], errors="coerce")
    df = df.loc[df[spec.lon_col].notna() & df[spec.lat_col].notna()].copy()

    if df.empty:
        print(f"[WARN] No valid coordinates for {spec.name}")
        return None

    if spec.basin_col in df.columns:
        df["basin_plot"] = df[spec.basin_col].astype(str)
    else:
        df["basin_plot"] = "UNKNOWN"

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[spec.lon_col], df[spec.lat_col]),
        crs="EPSG:4326",
    )
    gdf["dataset_name"] = spec.name

    for col in ["exclude_final", "inside_mainland_polygon", "special_mask_hit", "exclude_near_mainland"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].fillna(False).astype(bool)

    if "local_dist_km" in gdf.columns:
        gdf["local_dist_km"] = pd.to_numeric(gdf["local_dist_km"], errors="coerce")

    return gdf


# ==========================================================
# PLOTTING HELPERS
# ==========================================================

def add_land_sea_background(ax, mainland_plot, region_bbox=None):
    ax.set_facecolor(SEA_FACE)

    if mainland_plot is None or mainland_plot.empty:
        return

    sub = mainland_plot
    if region_bbox is not None:
        lon_min, lon_max, lat_min, lat_max = region_bbox
        sub = sub.loc[sub.intersects(box(lon_min, lat_min, lon_max, lat_max))].copy()

    if not sub.empty:
        sub.plot(ax=ax, facecolor=LAND_FACE, edgecolor="none", aspect="auto", zorder=0)


def add_river_line_overlay(ax, river_lines: gpd.GeoDataFrame, region_bbox=None, linewidth=0.7, alpha=0.6):
    if not SHOW_RIVER_LINES or river_lines is None or river_lines.empty:
        return

    rr = river_lines
    if region_bbox is not None:
        lon_min, lon_max, lat_min, lat_max = region_bbox
        bbox_geom = box(lon_min, lat_min, lon_max, lat_max)
        rr = rr.loc[rr.intersects(bbox_geom)].copy()

    if rr.empty:
        return

    rr.plot(ax=ax, color="deepskyblue", linewidth=linewidth, alpha=alpha, aspect="auto", zorder=8)


def add_circle_centers_and_labels(ax, circle_centers: gpd.GeoDataFrame, show_labels: bool) -> None:
    if circle_centers.empty or not SHOW_CIRCLE_CENTERS:
        return

    xs = circle_centers.geometry.x.values
    ys = circle_centers.geometry.y.values

    ax.scatter(
        xs,
        ys,
        s=CIRCLE_CENTER_MARKER_SIZE,
        marker="x",
        c="red",
        linewidths=1.8,
        zorder=30,
        label="circle mask centers",
    )

    if not show_labels:
        return

    for _, row in circle_centers.iterrows():
        name = row["name"]
        x = row.geometry.x
        y = row.geometry.y
        dx, dy = CIRCLE_LABEL_OFFSETS.get(name, (0.12, 0.08))

        ax.text(
            x + dx,
            y + dy,
            name,
            fontsize=CIRCLE_LABEL_SIZE,
            ha="left",
            va="center",
            color="red",
            zorder=31,
        )

        ax.plot(
            [x, x + dx],
            [y, y + dy],
            color="red",
            linewidth=CIRCLE_LABEL_CONNECTOR_WIDTH,
            zorder=29,
        )


def add_box_labels(ax, box_plot: gpd.GeoDataFrame, region_bbox=None):
    if box_plot is None or box_plot.empty:
        return

    sub = box_plot
    if region_bbox is not None:
        lon_min, lon_max, lat_min, lat_max = region_bbox
        sub = sub.loc[sub.intersects(box(lon_min, lat_min, lon_max, lat_max))].copy()

    if sub.empty:
        return

    for _, row in sub.iterrows():
        label = row.get("label", row.get("name", ""))
        rp = row.geometry.representative_point()
        dx, dy = BOX_LABEL_OFFSETS.get(label, (0.0, 0.0))
        ax.text(
            rp.x + dx,
            rp.y + dy,
            label,
            fontsize=8,
            color="dodgerblue",
            ha="center",
            va="center",
            zorder=26,
        )


def add_actual_mask_layers(
    ax,
    mainland_plot,
    coast_buffer_plot,
    circle_mask_plot,
    box_mask_plot,
    river_lines=None,
    region_bbox=None,
):
    add_land_sea_background(ax, mainland_plot, region_bbox=region_bbox)

    if river_lines is not None:
        add_river_line_overlay(ax, river_lines, region_bbox=region_bbox, linewidth=0.7, alpha=0.55)

    if coast_buffer_plot is not None and not coast_buffer_plot.empty:
        sub = coast_buffer_plot
        if region_bbox is not None:
            lon_min, lon_max, lat_min, lat_max = region_bbox
            sub = sub.loc[sub.intersects(box(lon_min, lat_min, lon_max, lat_max))].copy()

        if not sub.empty:
            sub.plot(
                ax=ax,
                alpha=0.08,
                edgecolor="none",
                color="lightsteelblue",
                aspect="auto",
                zorder=2,
            )

    if circle_mask_plot is not None and not circle_mask_plot.empty:
        sub = circle_mask_plot
        if region_bbox is not None:
            lon_min, lon_max, lat_min, lat_max = region_bbox
            sub = sub.loc[sub.intersects(box(lon_min, lat_min, lon_max, lat_max))].copy()

        if not sub.empty:
            sub.boundary.plot(
                ax=ax,
                color="red",
                linewidth=1.4,
                aspect="auto",
                zorder=10,
            )

    if box_mask_plot is not None and not box_mask_plot.empty:
        sub = box_mask_plot
        if region_bbox is not None:
            lon_min, lon_max, lat_min, lat_max = region_bbox
            sub = sub.loc[sub.intersects(box(lon_min, lat_min, lon_max, lat_max))].copy()

        if not sub.empty:
            sub.boundary.plot(
                ax=ax,
                color="dodgerblue",
                linewidth=1.6,
                linestyle="--",
                aspect="auto",
                zorder=11,
            )

    if mainland_plot is not None and not mainland_plot.empty:
        sub = mainland_plot
        if region_bbox is not None:
            lon_min, lon_max, lat_min, lat_max = region_bbox
            sub = sub.loc[sub.intersects(box(lon_min, lat_min, lon_max, lat_max))].copy()

        if not sub.empty:
            sub.boundary.plot(
                ax=ax,
                linewidth=0.8,
                color=COAST_COLOR,
                aspect="auto",
                zorder=12,
            )


def split_point_categories(gdf_plot: gpd.GeoDataFrame):
    empty = gdf_plot.iloc[0:0].copy()

    kept = empty.copy()
    mainland_2km = empty.copy()
    dj1_1km = empty.copy()
    special = empty.copy()
    inland = empty.copy()

    if gdf_plot is None or gdf_plot.empty:
        return kept, mainland_2km, dj1_1km, special, inland

    if "inside_mainland_polygon" in gdf_plot.columns:
        inland = gdf_plot.loc[gdf_plot["inside_mainland_polygon"]].copy()

    needed = {"exclude_final", "exclude_near_mainland", "local_dist_km", "special_mask_hit", "inside_mainland_polygon"}
    if not needed.issubset(gdf_plot.columns):
        kept = gdf_plot.copy()
        return kept, mainland_2km, dj1_1km, special, inland

    mainland_2km = gdf_plot.loc[
        (gdf_plot["exclude_final"]) &
        (~gdf_plot["inside_mainland_polygon"]) &
        (gdf_plot["exclude_near_mainland"]) &
        (~gdf_plot["special_mask_hit"]) &
        (np.isclose(gdf_plot["local_dist_km"], 2.0, equal_nan=False))
    ].copy()

    dj1_1km = gdf_plot.loc[
        (gdf_plot["exclude_final"]) &
        (~gdf_plot["inside_mainland_polygon"]) &
        (gdf_plot["exclude_near_mainland"]) &
        (~gdf_plot["special_mask_hit"]) &
        (np.isclose(gdf_plot["local_dist_km"], 1.0, equal_nan=False))
    ].copy()

    special = gdf_plot.loc[
        (gdf_plot["exclude_final"]) &
        (~gdf_plot["inside_mainland_polygon"]) &
        (gdf_plot["special_mask_hit"])
    ].copy()

    used_idx = mainland_2km.index.union(dj1_1km.index).union(special.index).union(inland.index)
    kept = gdf_plot.drop(index=used_idx, errors="ignore").copy()

    if "exclude_final" in kept.columns:
        kept = kept.loc[~kept["exclude_final"]].copy()

    return kept, mainland_2km, dj1_1km, special, inland


def add_point_layers(
    ax,
    gdf_plot,
    kept_size: float,
    mainland_2km_size: float,
    dj1_1km_size: float,
    special_size: float,
    inland_size: float,
):
    if gdf_plot is None or gdf_plot.empty:
        return

    kept, mainland_2km, dj1_1km, special, inland = split_point_categories(gdf_plot)

    if not kept.empty:
        kept.plot(
            ax=ax,
            markersize=kept_size,
            color=KEPT_COLOR,
            label="kept",
            aspect="auto",
            zorder=21,
        )

    if SHOW_MAINLAND_2KM_POINTS and not mainland_2km.empty:
        mainland_2km.plot(
            ax=ax,
            markersize=mainland_2km_size,
            marker=MAINLAND_2KM_MARKER,
            color=MAINLAND_2KM_COLOR,
            label="excluded: 2 km coast",
            aspect="auto",
            zorder=22,
        )

    if SHOW_DJ1_1KM_POINTS and not dj1_1km.empty:
        dj1_1km.plot(
            ax=ax,
            markersize=dj1_1km_size,
            marker=DJ1_1KM_MARKER,
            color=DJ1_1KM_COLOR,
            label="excluded: DJ1 1 km coast",
            aspect="auto",
            zorder=23,
        )

    if SHOW_SPECIAL_MASK_POINTS and not special.empty:
        special.plot(
            ax=ax,
            markersize=special_size,
            marker=SPECIAL_MASK_MARKER,
            color=SPECIAL_MASK_COLOR,
            label="excluded: special mask",
            aspect="auto",
            zorder=24,
        )

    if SHOW_INLAND_POINTS and not inland.empty:
        inland.plot(
            ax=ax,
            markersize=inland_size,
            marker=INLAND_MARKER,
            color=INLAND_COLOR,
            label="inside mainland polygon",
            aspect="auto",
            zorder=25,
        )


def make_bbox_from_points(gdf: gpd.GeoDataFrame, pad_deg: float):
    xmin = gdf.geometry.x.min()
    xmax = gdf.geometry.x.max()
    ymin = gdf.geometry.y.min()
    ymax = gdf.geometry.y.max()

    if np.isclose(xmin, xmax):
        xmin -= 0.05
        xmax += 0.05
    if np.isclose(ymin, ymax):
        ymin -= 0.05
        ymax += 0.05

    return (xmin - pad_deg, xmax + pad_deg, ymin - pad_deg, ymax + pad_deg)


# ==========================================================
# MAIN FIGURES
# ==========================================================

def plot_dataset_panel(
    ax,
    name,
    gdf,
    mainland_plot,
    coast_buffer_plot,
    circle_mask_plot,
    box_mask_plot,
    river_lines,
):
    gdf_plot = maybe_downsample_for_plot(gdf, MAX_POINTS_PER_DATASET)
    bbox = make_bbox_from_points(gdf_plot, QC_PAD_DEG)

    add_actual_mask_layers(
        ax,
        mainland_plot,
        coast_buffer_plot,
        circle_mask_plot,
        box_mask_plot,
        river_lines=river_lines,
        region_bbox=bbox,
    )

    add_point_layers(
        ax,
        gdf_plot,
        kept_size=KEPT_SIZE,
        mainland_2km_size=MAINLAND_2KM_SIZE,
        dj1_1km_size=DJ1_1KM_SIZE,
        special_size=SPECIAL_MASK_SIZE,
        inland_size=INLAND_SIZE,
    )

    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")


def plot_panel_2x3(
    out_path,
    dataset_gdfs,
    mainland_plot,
    coast_buffer_plot,
    circle_mask_plot,
    box_mask_plot,
    river_lines,
):
    fig, axes = plt.subplots(2, 3, figsize=PANEL_FIGSIZE)
    axes = axes.ravel()

    for ax, (name, gdf) in zip(axes, dataset_gdfs):
        plot_dataset_panel(
            ax,
            name,
            gdf,
            mainland_plot,
            coast_buffer_plot,
            circle_mask_plot,
            box_mask_plot,
            river_lines,
        )

    for j in range(len(dataset_gdfs), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        uniq = dict(zip(labels, handles))
        fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=5, frameon=True)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OUTPUT] {out_path}")


def plot_combined_med(
    out_path,
    dataset_gdfs,
    mainland_plot,
    coast_buffer_plot,
    circle_mask_plot,
    box_mask_plot,
    circle_centers,
    river_lines,
):
    all_gdf = pd.concat([g for _, g in dataset_gdfs], ignore_index=True)
    all_gdf = gpd.GeoDataFrame(all_gdf, geometry="geometry", crs="EPSG:4326")
    all_gdf = maybe_downsample_for_plot(all_gdf, MAX_POINTS_PER_DATASET)

    fig, ax = plt.subplots(figsize=COMBINED_FIGSIZE)

    med_bbox = (MEDITERRANEAN_BBOX[0], MEDITERRANEAN_BBOX[2], MEDITERRANEAN_BBOX[1], MEDITERRANEAN_BBOX[3])

    add_actual_mask_layers(
        ax,
        mainland_plot,
        coast_buffer_plot,
        circle_mask_plot,
        box_mask_plot,
        river_lines=river_lines,
        region_bbox=med_bbox,
    )

    add_point_layers(
        ax,
        all_gdf,
        kept_size=KEPT_SIZE,
        mainland_2km_size=MAINLAND_2KM_SIZE,
        dj1_1km_size=DJ1_1KM_SIZE,
        special_size=SPECIAL_MASK_SIZE,
        inland_size=INLAND_SIZE,
    )
    add_circle_centers_and_labels(ax, circle_centers, show_labels=SHOW_CIRCLE_LABELS_IN_COMBINED)

    if SHOW_BOX_LABELS_IN_COMBINED:
        add_box_labels(ax, box_mask_plot, region_bbox=med_bbox)

    ax.set_title("All datasets combined", fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(MEDITERRANEAN_BBOX[0], MEDITERRANEAN_BBOX[2])
    ax.set_ylim(MEDITERRANEAN_BBOX[1], MEDITERRANEAN_BBOX[3])
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys())

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OUTPUT] {out_path}")


# ==========================================================
# ZOOM FIGURES
# ==========================================================

def plot_zoom_regions(
    all_points: gpd.GeoDataFrame,
    mainland_plot: gpd.GeoDataFrame,
    coast_buffer_plot: gpd.GeoDataFrame,
    circle_mask_plot: gpd.GeoDataFrame,
    box_mask_plot: gpd.GeoDataFrame,
    circle_centers: gpd.GeoDataFrame,
    river_lines: gpd.GeoDataFrame,
) -> None:
    if not SAVE_ZOOM_PLOTS:
        return

    for region in ZOOM_REGIONS:
        lon_min = region["lon_min"]
        lon_max = region["lon_max"]
        lat_min = region["lat_min"]
        lat_max = region["lat_max"]
        target = region["name"]

        pts = all_points.loc[
            (all_points.geometry.x >= lon_min) &
            (all_points.geometry.x <= lon_max) &
            (all_points.geometry.y >= lat_min) &
            (all_points.geometry.y <= lat_max)
        ].copy()

        bbox_geom = box(lon_min, lat_min, lon_max, lat_max)

        circle_mask_sub = circle_mask_plot.loc[circle_mask_plot.intersects(bbox_geom)].copy()
        box_mask_sub = box_mask_plot.loc[box_mask_plot.intersects(bbox_geom)].copy()
        circle_centers_sub = circle_centers.loc[
            (circle_centers.geometry.x >= lon_min) &
            (circle_centers.geometry.x <= lon_max) &
            (circle_centers.geometry.y >= lat_min) &
            (circle_centers.geometry.y <= lat_max)
        ].copy()

        fig, ax = plt.subplots(figsize=ZOOM_FIGSIZE)

        add_actual_mask_layers(
            ax,
            mainland_plot,
            coast_buffer_plot,
            circle_mask_sub,
            box_mask_sub,
            river_lines=river_lines,
            region_bbox=(lon_min, lon_max, lat_min, lat_max),
        )

        if not pts.empty:
            add_point_layers(
                ax,
                pts,
                kept_size=ZOOM_POINT_SIZE_KEPT,
                mainland_2km_size=ZOOM_MAINLAND_2KM_SIZE,
                dj1_1km_size=ZOOM_DJ1_1KM_SIZE,
                special_size=ZOOM_SPECIAL_MASK_SIZE,
                inland_size=ZOOM_INLAND_SIZE,
            )

        add_circle_centers_and_labels(ax, circle_centers_sub, show_labels=SHOW_CIRCLE_LABELS_IN_ZOOMS)

        if SHOW_BOX_LABELS_IN_ZOOMS:
            add_box_labels(ax, box_mask_sub, region_bbox=(lon_min, lon_max, lat_min, lat_max))

        ax.set_title(target, fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            uniq = dict(zip(labels, handles))
            ax.legend(uniq.values(), uniq.keys())

        out = ZOOM_OUTPUT_DIR / f"{target}_zoom.png"
        fig.tight_layout()
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"[OUTPUT] {out}")


# ==========================================================
# BASIN MAPS
# ==========================================================

def plot_single_basin_map(
    out_path: Path,
    basin_title: str,
    basin_gdf: gpd.GeoDataFrame,
    mainland_plot: gpd.GeoDataFrame,
    coast_buffer_plot: gpd.GeoDataFrame,
    circle_mask_plot: gpd.GeoDataFrame,
    box_mask_plot: gpd.GeoDataFrame,
    circle_centers: gpd.GeoDataFrame,
    river_lines: gpd.GeoDataFrame,
    pad_deg: float = BASIN_PAD_DEG,
):
    if basin_gdf.empty:
        return

    basin_gdf = maybe_downsample_for_plot(basin_gdf, MAX_POINTS_PER_BASIN)
    bbox = make_bbox_from_points(basin_gdf, pad_deg)
    bbox_geom = box(bbox[0], bbox[2], bbox[1], bbox[3])

    circle_centers_sub = circle_centers.loc[
        (circle_centers.geometry.x >= bbox[0]) &
        (circle_centers.geometry.x <= bbox[1]) &
        (circle_centers.geometry.y >= bbox[2]) &
        (circle_centers.geometry.y <= bbox[3])
    ].copy()

    circle_mask_sub = circle_mask_plot.loc[circle_mask_plot.intersects(bbox_geom)].copy()
    box_mask_sub = box_mask_plot.loc[box_mask_plot.intersects(bbox_geom)].copy()

    fig, ax = plt.subplots(figsize=BASIN_FIGSIZE)

    add_actual_mask_layers(
        ax,
        mainland_plot,
        coast_buffer_plot,
        circle_mask_sub,
        box_mask_sub,
        river_lines=river_lines,
        region_bbox=bbox,
    )

    add_point_layers(
        ax,
        basin_gdf,
        kept_size=BASIN_POINT_SIZE_KEPT,
        mainland_2km_size=BASIN_MAINLAND_2KM_SIZE,
        dj1_1km_size=BASIN_DJ1_1KM_SIZE,
        special_size=BASIN_SPECIAL_MASK_SIZE,
        inland_size=BASIN_INLAND_SIZE,
    )

    add_circle_centers_and_labels(ax, circle_centers_sub, show_labels=SHOW_CIRCLE_LABELS_IN_BASINS)

    if SHOW_BOX_LABELS_IN_BASINS:
        add_box_labels(ax, box_mask_sub, region_bbox=bbox)

    ax.set_title(basin_title, fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys())

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OUTPUT] {out_path}")


def save_atomic_basin_maps(
    dataset_gdfs,
    mainland_plot,
    coast_buffer_plot,
    circle_mask_plot,
    box_mask_plot,
    circle_centers,
    river_lines,
):
    all_gdf = pd.concat([g for _, g in dataset_gdfs], ignore_index=True)
    all_gdf = gpd.GeoDataFrame(all_gdf, geometry="geometry", crs="EPSG:4326")

    if "basin_plot" not in all_gdf.columns:
        print("[WARN] No basin_plot column available for atomic basin maps.")
        return

    basins = sorted([b for b in all_gdf["basin_plot"].dropna().astype(str).unique() if b and b != "UNKNOWN"])

    for basin in basins:
        sub = all_gdf.loc[all_gdf["basin_plot"].astype(str) == basin].copy()
        if sub.empty:
            continue

        out = ATOMIC_OUTPUT_DIR / f"{basin}.png"
        plot_single_basin_map(
            out_path=out,
            basin_title=basin,
            basin_gdf=sub,
            mainland_plot=mainland_plot,
            coast_buffer_plot=coast_buffer_plot,
            circle_mask_plot=circle_mask_plot,
            box_mask_plot=box_mask_plot,
            circle_centers=circle_centers,
            river_lines=river_lines,
        )


def save_composed_basin_maps(
    dataset_gdfs,
    mainland_plot,
    coast_buffer_plot,
    circle_mask_plot,
    box_mask_plot,
    circle_centers,
    river_lines,
):
    all_gdf = pd.concat([g for _, g in dataset_gdfs], ignore_index=True)
    all_gdf = gpd.GeoDataFrame(all_gdf, geometry="geometry", crs="EPSG:4326")

    if "basin_plot" not in all_gdf.columns:
        print("[WARN] No basin_plot column available for composed basin maps.")
        return

    for comp_code, comp_def in COMPOSED_BASINS.items():
        members = list(comp_def["members"])
        title = f"{comp_code} – {comp_def['name']}"
        sub = all_gdf.loc[all_gdf["basin_plot"].isin(members)].copy()

        if sub.empty:
            continue

        out = COMPOSED_OUTPUT_DIR / f"{comp_code}.png"
        plot_single_basin_map(
            out_path=out,
            basin_title=title,
            basin_gdf=sub,
            mainland_plot=mainland_plot,
            coast_buffer_plot=coast_buffer_plot,
            circle_mask_plot=circle_mask_plot,
            box_mask_plot=box_mask_plot,
            circle_centers=circle_centers,
            river_lines=river_lines,
        )


# ==========================================================
# MAIN
# ==========================================================

def main():
    mainland = build_mainland_polygons()
    mainland_plot = clean_for_plot(mainland, to_crs="EPSG:4326")

    circle_mask_plot = build_circle_mask_geometries()
    box_mask_plot = build_box_mask_geometries()
    circle_centers = build_circle_center_points()
    river_lines = build_river_lines()

    coast_buffer = mainland.copy()
    coast_buffer["geometry"] = coast_buffer.geometry.buffer(DEFAULT_DIST_KM * 1000.0)
    coast_buffer_plot = clean_for_plot(coast_buffer, to_crs="EPSG:4326")

    print("[DEBUG] circle mask bounds :", circle_mask_plot.total_bounds if len(circle_mask_plot) else "empty")
    print("[DEBUG] box mask bounds    :", box_mask_plot.total_bounds if len(box_mask_plot) else "empty")

    dataset_gdfs = []
    for spec in DATASETS:
        gdf = load_diag_as_gdf(spec)
        if gdf is None:
            continue
        dataset_gdfs.append((spec.name, gdf))

    if not dataset_gdfs:
        print("[WARN] No diagnostics datasets could be loaded.")
        return

    plot_panel_2x3(
        OUT_PANEL,
        dataset_gdfs,
        mainland_plot,
        coast_buffer_plot,
        circle_mask_plot,
        box_mask_plot,
        river_lines,
    )

    plot_combined_med(
        OUT_COMBINED,
        dataset_gdfs,
        mainland_plot,
        coast_buffer_plot,
        circle_mask_plot,
        box_mask_plot,
        circle_centers,
        river_lines,
    )

    all_points = pd.concat([g for _, g in dataset_gdfs], ignore_index=True)
    all_points = gpd.GeoDataFrame(all_points, geometry="geometry", crs="EPSG:4326")

    plot_zoom_regions(
        all_points,
        mainland_plot,
        coast_buffer_plot,
        circle_mask_plot,
        box_mask_plot,
        circle_centers,
        river_lines,
    )

    if SAVE_BASIN_MAPS:
        save_atomic_basin_maps(
            dataset_gdfs,
            mainland_plot,
            coast_buffer_plot,
            circle_mask_plot,
            box_mask_plot,
            circle_centers,
            river_lines,
        )

        save_composed_basin_maps(
            dataset_gdfs,
            mainland_plot,
            coast_buffer_plot,
            circle_mask_plot,
            box_mask_plot,
            circle_centers,
            river_lines,
        )

    print("[DONE] Finished plotting all datasets.")


if __name__ == "__main__":
    main()