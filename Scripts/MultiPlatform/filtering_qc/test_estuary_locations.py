#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
River/strait mask tuning plots using MapPlotter satellite basemap.

Purpose
-------
For each test group:
- plot nearby stations on a MapPlotter satellite background
- draw candidate circle and/or box masks
- show which stations fall inside candidate masks
- also highlight:
    * inland points
    * points within COAST_TEST_KM of mainland coastline

Outputs
-------
One PNG per target group in:
    /Users/elena/Projects/IRB/MedClim/Plots/RiverMaskTuning_MapPlotter

Optional:
    river_mask_candidates_overview_satellite.png

Legend
------
- Green points   = outside candidate mask(s)
- Black points   = inside candidate mask(s)
- Orange points  = within COAST_TEST_KM of coastline
- Magenta X      = inland points
- Red X          = candidate mouth location
- Red circle     = candidate circle mask
- Dodgerblue box = candidate rectangular mask
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import MapPlotter as mp

from pyproj import Geod
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


# ==========================================================
# OUTPUT
# ==========================================================

OUTPUT_DIR = Path("/Users/elena/Projects/IRB/MedClim/Plots/RiverMaskTuning_MapPlotter")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_COMBINED_OVERVIEW = True
OUT_COMBINED = OUTPUT_DIR / "river_mask_candidates_overview_satellite.png"


# ==========================================================
# MAP / STYLE
# ==========================================================

POINT_SIZE_OUTSIDE = 10
POINT_SIZE_INSIDE = 14
POINT_SIZE_COAST = 16
POINT_SIZE_INLAND = 28
POINT_ALPHA = 0.85

RIVER_MARKER_SIZE = 55
RIVER_LABEL_SIZE = 9
RIVER_LINEWIDTH = 1.8
BOX_LINEWIDTH = 1.8

ZOOM_FIGSIZE = (8.5, 7.5)
COMBINED_FIGSIZE = (13, 10)

SHOW_LABELS = True
MAX_POINTS_PER_ZOOM = 50000

MAP_ZOOM_DEFAULT = 10
MAP_ZOOM_OVERVIEW = 5

MEDITERRANEAN_XLIM = [-12.0, 42.5]
MEDITERRANEAN_YLIM = [18.0, 52.0]

METRIC_CRS = "EPSG:3035"
MEDITERRANEAN_BBOX = (-12.0, 18.0, 42.5, 52.0)
MAINLAND_MIN_AREA_KM2 = 50000.0
MAINLAND_MIN_AREA_M2 = MAINLAND_MIN_AREA_KM2 * 1e6

# Test coastline proximity threshold
COAST_TEST_KM = 2.0

CACHE_DIR = Path.home() / ".cache" / "medclim_mainland_mask"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
NE_LAND_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip"


# ==========================================================
# DATASETS
# ==========================================================

@dataclass
class PointSpec:
    name: str
    path: Path
    lon_col: str
    lat_col: str


DATASETS: List[PointSpec] = [
    PointSpec(
        name="SDN",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Seadatanet/Seadatanet.profiles_only_with_basins_mainland_special_excluded_profile_diagnostics.csv"),
        lon_col="Longitude [degrees_east]",
        lat_col="Latitude [degrees_north]",
    ),
    PointSpec(
        name="ARGO",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Argo/CSV/Argo_with_basins_mainland_special_excluded_profile_diagnostics.csv"),
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
    ),
    PointSpec(
        name="CTD_HR_2019_2023_MFSD",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/CTD_HR_2019_2023_MFSD_long_fixed_mainland_special_excluded_profile_diagnostics.csv"),
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
    ),
    PointSpec(
        name="RB_Northern_Adriatic_Transect_1979_2018",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/RB_Northern_Adriatic_Transect_1979_2018_long_fixed_mainland_special_excluded_profile_diagnostics.csv"),
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
    ),
    PointSpec(
        name="Palagruza_Transect_1952_2018",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/Palagruza_Transect_1952_2018_long_fixed_mainland_special_excluded_profile_diagnostics.csv"),
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
    ),
    PointSpec(
        name="IOR_Adriatic_database_1987_2019",
        path=Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/Processed/IOR_Adriatic_database_1987_2019_long_fixed_mainland_special_excluded_profile_diagnostics.csv"),
        lon_col="LONGITUDE",
        lat_col="LATITUDE",
    ),
]


# ==========================================================
# CANDIDATE TESTS
#
# Each candidate must have:
#   type = "circle" or "box"
#
# Circle:
#   {"type":"circle","name":"...", "lon":..., "lat":..., "radius_km":...}
#
# Box:
#   {"type":"box","name":"...", "lon_min":..., "lon_max":..., "lat_min":..., "lat_max":...}
# ==========================================================

CANDIDATE_TESTS = [
    {
        "group": "Rhone",
        "candidates": [
            {"type": "circle", "name": "Rhone", "lon": 4.80, "lat": 43.35, "radius_km": 20.0},
        ],
        "zoom_pad_deg": 0.8,
        "map_zoom": 9,
    },
    {
        "group": "Ebro",
        "candidates": [
            {"type": "circle", "name": "Ebro", "lon": 0.83, "lat": 40.72, "radius_km": 20.0},
        ],
        "zoom_pad_deg": 0.7,
        "map_zoom": 10,
    },
        {
        "group": "Po",
        "candidates": [
            {
                "type": "box",
                "name": "Po_delta_box",
                "lon_min": 12.2,
                "lon_max": 12.7,
                "lat_min": 44.7,
                "lat_max": 45.15,
            },
        ],
        "zoom_pad_deg": 0.40,
        "map_zoom": 11,
    },
    {
        "group": "Nile",
        "candidates": [
            {"type": "circle", "name": "Nile_Rosetta",  "lon": 30.36702, "lat": 31.46485, "radius_km": 20.0},
            {"type": "circle", "name": "Nile_Damietta", "lon": 31.84372, "lat": 31.52178, "radius_km": 20.0},
        ],
        "zoom_pad_deg": 1.8,
        "map_zoom": 8,
    },
    {
        "group": "Drin_Bojana",
        "candidates": [
            {"type": "circle", "name": "Drin_Bojana", "lon": 19.30, "lat": 41.85, "radius_km": 10.0},
        ],
        "zoom_pad_deg": 0.45,
        "map_zoom": 11,
    },
    {
        "group": "Neretva",
        "candidates": [
            {
                "type": "box",
                "name": "Neretva_1",
                "lon_min": 17.30,
                "lon_max": 17.52,
                "lat_min": 42.92,
                "lat_max": 43.08,
            },
            {
                "type": "box",
                "name": "Neretva_2",
                "lon_min": 17.45,
                "lon_max": 17.62,
                "lat_min": 42.88,
                "lat_max": 42.97,
            },
            {
                "type": "box",
                "name": "Neretva_3",
                "lon_min": 17.62,
                "lon_max": 17.78,
                "lat_min": 42.80,
                "lat_max": 42.88,
            },
        ],
        "zoom_pad_deg": 0.30,
        "map_zoom": 12,
    },
#   {
#       "group": "Neretva",
#       "candidates": [
#           {
#               "type": "box",
#               "name": "Neretva_box",
#               "lon_min": 17.33,
#               "lon_max": 17.50,
#               "lat_min": 42.96,
#               "lat_max": 43.05,
#           },
#       ],
#       "zoom_pad_deg": 0.22,
#       "map_zoom": 12,
#   },
    {
        "group": "Tiber",
        "candidates": [
            {"type": "circle", "name": "Tiber", "lon": 12.23, "lat": 41.74, "radius_km": 10.0},
        ],
        "zoom_pad_deg": 0.45,
        "map_zoom": 11,
    },
    {
        "group": "Adige",
        "candidates": [
            {"type": "circle", "name": "Adige_r8", "lon": 12.33, "lat": 45.16, "radius_km": 10.0},
        ],
        "zoom_pad_deg": 0.40,
        "map_zoom": 11,  
    },
    {
        "group": "Seyhan",
        "candidates": [
            {"type": "circle", "name": "Seyhan", "lon": 34.91, "lat": 36.73, "radius_km": 12.0},
        ],
        "zoom_pad_deg": 0.55,
        "map_zoom": 10,
    },
    {
        "group": "Ceyhan",
        "candidates": [
            {"type": "circle", "name": "Ceyhan", "lon": 35.55, "lat": 36.56, "radius_km": 12.0},
        ],
        "zoom_pad_deg": 0.45,
        "map_zoom": 11,
    },
    {
        "group": "Krka",
        "candidates": [
            {
                "type": "box",
                "name": "Krka_box",
                "lon_min": 15.73,
                "lon_max": 15.90,
                "lat_min": 43.65,
                "lat_max": 43.76,
            },
        ],
        "zoom_pad_deg": 0.18,
        "map_zoom": 12,
    },
    {
        "group": "Zrmanja",
        "candidates": [
            {
                "type": "box",
                "name": "Zrmanja_box",
                "lon_min": 15.3,
                "lon_max": 15.65,
                "lat_min": 44.1,
                "lat_max": 44.4,
            },
        ],
        "zoom_pad_deg": 0.20,
        "map_zoom": 12,
    },
    {
        "group": "Dardanelles",
        "candidates": [
            {"type": "box", "name": "Dardanelles_box", "lon_min": 26.10, "lon_max": 27.15, "lat_min": 39.98, "lat_max": 40.55},
        ],
        "zoom_pad_deg": 0.70,
        "map_zoom": 10,
    },
]


# ==========================================================
# HELPERS
# ==========================================================

def maybe_downsample_df(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = max(1, math.ceil(len(df) / max_points))
    return df.iloc[::step].copy()


def load_points_df(spec: PointSpec) -> pd.DataFrame | None:
    if not spec.path.exists():
        print(f"[WARN] Missing file for {spec.name}: {spec.path}")
        return None

    df = pd.read_csv(spec.path, low_memory=False)

    if spec.lon_col not in df.columns or spec.lat_col not in df.columns:
        print(f"[WARN] Missing lon/lat columns in {spec.name}")
        return None

    df[spec.lon_col] = pd.to_numeric(df[spec.lon_col], errors="coerce")
    df[spec.lat_col] = pd.to_numeric(df[spec.lat_col], errors="coerce")
    df = df.loc[df[spec.lon_col].notna() & df[spec.lat_col].notna()].copy()

    if df.empty:
        print(f"[WARN] No valid coordinates in {spec.name}")
        return None

    return pd.DataFrame({
        "lon": df[spec.lon_col].to_numpy(dtype=float),
        "lat": df[spec.lat_col].to_numpy(dtype=float),
        "dataset_name": spec.name,
    })


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
    gg = gg.loc[np.isfinite(bounds).all(axis=1)].copy()
    if len(gg) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=to_crs)

    return gg


def ensure_natural_earth_land() -> Path:
    zip_path = CACHE_DIR / Path(NE_LAND_URL).name
    if not zip_path.exists():
        urllib.request.urlretrieve(NE_LAND_URL, zip_path)
    return zip_path


def build_mainland_and_coastline():
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
    coastline = mainland_union.boundary

    mainland_gdf = gpd.GeoDataFrame({"name": ["mainland"], "geometry": [mainland_union]}, crs=METRIC_CRS)
    coastline_gdf = gpd.GeoDataFrame({"name": ["coastline"], "geometry": [coastline]}, crs=METRIC_CRS)

    return mainland_gdf, coastline_gdf


def make_geodesic_circle_polygon(lon: float, lat: float, radius_km: float, n_points: int = 240) -> Polygon:
    geod = Geod(ellps="WGS84")
    azimuths = np.linspace(0, 360, n_points, endpoint=False)
    coords = []
    for az in azimuths:
        lon2, lat2, _ = geod.fwd(lon, lat, az, radius_km * 1000.0)
        coords.append((lon2, lat2))
    coords.append(coords[0])
    return Polygon(coords)


def build_candidate_circle_gdf(candidates: list[dict]) -> gpd.GeoDataFrame:
    rows = []
    for c in candidates:
        if c["type"] != "circle":
            continue
        rows.append({
            "name": c["name"],
            "lon": c["lon"],
            "lat": c["lat"],
            "radius_km": c["radius_km"],
            "geometry": make_geodesic_circle_polygon(c["lon"], c["lat"], c["radius_km"]),
        })
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def build_candidate_box_gdf(candidates: list[dict]) -> gpd.GeoDataFrame:
    rows = []
    for c in candidates:
        if c["type"] != "box":
            continue
        rows.append({
            "name": c["name"],
            "geometry": box(c["lon_min"], c["lat_min"], c["lon_max"], c["lat_max"]),
        })
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def build_candidate_points_df(candidates: list[dict]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        if c["type"] != "circle":
            continue
        rows.append({
            "name": c["name"],
            "lon": c["lon"],
            "lat": c["lat"],
            "radius_km": c["radius_km"],
        })
    return pd.DataFrame(rows)


def classify_points(points_df: pd.DataFrame,
                    circle_gdf: gpd.GeoDataFrame,
                    box_gdf: gpd.GeoDataFrame,
                    mainland_gdf: gpd.GeoDataFrame,
                    coastline_gdf: gpd.GeoDataFrame,
                    coast_test_km: float) -> pd.DataFrame:
    out = points_df.copy()
    out["inside_candidate"] = False
    out["inside_mainland"] = False
    out["within_coast_test"] = False
    out["dist_to_coast_km"] = np.nan

    if out.empty:
        return out

    pts_gdf = gpd.GeoDataFrame(
        out.copy(),
        geometry=gpd.points_from_xy(out["lon"], out["lat"]),
        crs="EPSG:4326",
    )

    # candidate masks in lon/lat
    hit = np.zeros(len(pts_gdf), dtype=bool)

    if len(circle_gdf) > 0:
        for geom in circle_gdf.geometry:
            hit |= pts_gdf.within(geom).to_numpy()

    if len(box_gdf) > 0:
        for geom in box_gdf.geometry:
            hit |= pts_gdf.within(geom).to_numpy()

    out["inside_candidate"] = hit

    # inland + coast distance in metric CRS
    pts_metric = pts_gdf.to_crs(METRIC_CRS)
    mainland_union = mainland_gdf.geometry.iloc[0]
    coastline = coastline_gdf.geometry.iloc[0]

    out["inside_mainland"] = pts_metric.within(mainland_union).to_numpy()

    dist_m = pts_metric.geometry.distance(coastline).to_numpy(dtype=float)
    out["dist_to_coast_km"] = dist_m / 1000.0
    out["within_coast_test"] = dist_m <= (coast_test_km * 1000.0)

    return out


def format_candidate_text(candidates: list[dict], coast_test_km: float) -> str:
    lines = [f"coast test = {coast_test_km:.1f} km"]
    for c in candidates:
        if c["type"] == "circle":
            lines.append(f"{c['name']}: ({c['lon']:.3f}, {c['lat']:.3f}), r={c['radius_km']:.1f} km")
        elif c["type"] == "box":
            lines.append(
                f"{c['name']}: "
                f"lon[{c['lon_min']:.3f}, {c['lon_max']:.3f}], "
                f"lat[{c['lat_min']:.3f}, {c['lat_max']:.3f}]"
            )
    return "\n".join(lines)


def infer_group_center_and_bbox(candidates: list[dict], zoom_pad_deg: float):
    xs = []
    ys = []

    for c in candidates:
        if c["type"] == "circle":
            xs.append(c["lon"])
            ys.append(c["lat"])
        elif c["type"] == "box":
            xs.extend([c["lon_min"], c["lon_max"]])
            ys.extend([c["lat_min"], c["lat_max"]])

    center_lon = float(np.mean(xs))
    center_lat = float(np.mean(ys))

    lon_min = min(xs) - zoom_pad_deg
    lon_max = max(xs) + zoom_pad_deg
    lat_min = min(ys) - zoom_pad_deg
    lat_max = max(ys) + zoom_pad_deg

    return center_lon, center_lat, lon_min, lon_max, lat_min, lat_max


def build_mapplotter_axes(xlim, ylim, figsize=(8, 7), map_zoom=10):
    plotter = mp.MapPlotter(projection="PlateCarree")

    params = plotter.defaultParams()
    params["dpi"] = 120
    params["xlim"] = list(xlim)
    params["ylim"] = list(ylim)
    params["features"] = ["continents", "tilemap"]
    params["map_zoom"] = map_zoom
    params["map_style"] = "satellite"
    params["xlabel"] = ["Longitude (deg)"]
    params["ylabel"] = ["Latitude (deg)"]
    params["draw_cbar"] = False

    plotter.plot_empty(params=params, clear=True)

    fig = plotter._fig
    ax = plotter._ax

    try:
        fig.set_size_inches(figsize[0], figsize[1], forward=True)
    except Exception:
        pass

    return plotter, fig, ax


# ==========================================================
# PLOTTING
# ==========================================================

def plot_single_group(group_name: str,
                      candidates: list[dict],
                      zoom_pad_deg: float,
                      map_zoom: int,
                      all_points: pd.DataFrame,
                      mainland_gdf: gpd.GeoDataFrame,
                      coastline_gdf: gpd.GeoDataFrame) -> None:
    _, _, lon_min, lon_max, lat_min, lat_max = infer_group_center_and_bbox(candidates, zoom_pad_deg)

    pts = all_points.loc[
        (all_points["lon"] >= lon_min) &
        (all_points["lon"] <= lon_max) &
        (all_points["lat"] >= lat_min) &
        (all_points["lat"] <= lat_max)
    ].copy()

    pts = maybe_downsample_df(pts, MAX_POINTS_PER_ZOOM)

    circles = build_candidate_circle_gdf(candidates)
    boxes = build_candidate_box_gdf(candidates)
    centers = build_candidate_points_df(candidates)

    pts = classify_points(
        pts,
        circle_gdf=circles,
        box_gdf=boxes,
        mainland_gdf=mainland_gdf,
        coastline_gdf=coastline_gdf,
        coast_test_km=COAST_TEST_KM,
    )

    plotter, fig, ax = build_mapplotter_axes(
        xlim=[lon_min, lon_max],
        ylim=[lat_min, lat_max],
        figsize=ZOOM_FIGSIZE,
        map_zoom=map_zoom,
    )

    if not pts.empty:
        inland = pts.loc[pts["inside_mainland"]].copy()
        coast = pts.loc[(~pts["inside_mainland"]) & pts["within_coast_test"]].copy()
        inside = pts.loc[(~pts["inside_mainland"]) & (~pts["within_coast_test"]) & pts["inside_candidate"]].copy()
        outside = pts.loc[(~pts["inside_mainland"]) & (~pts["within_coast_test"]) & (~pts["inside_candidate"])].copy()

        if not outside.empty:
            ax.scatter(
                outside["lon"].values,
                outside["lat"].values,
                s=POINT_SIZE_OUTSIDE,
                c="forestgreen",
                alpha=POINT_ALPHA,
                zorder=20,
                label="outside candidate mask",
            )

        if not inside.empty:
            ax.scatter(
                inside["lon"].values,
                inside["lat"].values,
                s=POINT_SIZE_INSIDE,
                c="black",
                alpha=POINT_ALPHA,
                zorder=21,
                label="inside candidate mask",
            )

        if not coast.empty:
            ax.scatter(
                coast["lon"].values,
                coast["lat"].values,
                s=POINT_SIZE_COAST,
                c="orange",
                alpha=POINT_ALPHA,
                zorder=22,
                label=f"within {COAST_TEST_KM:.1f} km coast",
            )

        if not inland.empty:
            ax.scatter(
                inland["lon"].values,
                inland["lat"].values,
                s=POINT_SIZE_INLAND,
                c="magenta",
                marker="x",
                linewidths=1.4,
                zorder=23,
                label="inside mainland polygon",
            )

    if len(circles) > 0:
        circles.boundary.plot(
            ax=ax,
            color="red",
            linewidth=RIVER_LINEWIDTH,
            zorder=30,
        )

    if len(boxes) > 0:
        boxes.boundary.plot(
            ax=ax,
            color="dodgerblue",
            linewidth=BOX_LINEWIDTH,
            linestyle="--",
            zorder=31,
        )

    if not centers.empty:
        ax.scatter(
            centers["lon"].values,
            centers["lat"].values,
            s=RIVER_MARKER_SIZE,
            marker="x",
            c="red",
            linewidths=1.8,
            zorder=32,
            label="candidate mouth",
        )

        if SHOW_LABELS:
            for _, row in centers.iterrows():
                txt = f"{row['name']}\nr={row['radius_km']:.1f}"
                ax.text(
                    row["lon"] + 0.015,
                    row["lat"] + 0.015,
                    txt,
                    fontsize=RIVER_LABEL_SIZE,
                    color="red",
                    ha="left",
                    va="bottom",
                    zorder=33,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                )

    if len(boxes) > 0 and SHOW_LABELS:
        for _, row in boxes.iterrows():
            rp = row.geometry.representative_point()
            ax.text(
                rp.x,
                rp.y,
                row["name"],
                fontsize=RIVER_LABEL_SIZE,
                color="dodgerblue",
                ha="center",
                va="center",
                zorder=33,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    ax.set_title(group_name, fontsize=12)

    summary = format_candidate_text(candidates, COAST_TEST_KM)
    ax.text(
        0.01,
        0.01,
        summary,
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.9),
        zorder=50,
    )

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper right", frameon=True)

    out = OUTPUT_DIR / f"{group_name}_candidate_test_satellite.png"
    fig.tight_layout()
    plotter.save(str(out), dpi=220)
    plt.close(fig)
    print(f"[OUTPUT] {out}")


def plot_combined_overview(tests: list[dict]) -> None:
    plotter, fig, ax = build_mapplotter_axes(
        xlim=MEDITERRANEAN_XLIM,
        ylim=MEDITERRANEAN_YLIM,
        figsize=COMBINED_FIGSIZE,
        map_zoom=MAP_ZOOM_OVERVIEW,
    )

    for test in tests:
        circles = build_candidate_circle_gdf(test["candidates"])
        boxes = build_candidate_box_gdf(test["candidates"])
        centers = build_candidate_points_df(test["candidates"])

        if len(circles) > 0:
            circles.boundary.plot(ax=ax, color="red", linewidth=1.1, zorder=20)

        if len(boxes) > 0:
            boxes.boundary.plot(ax=ax, color="dodgerblue", linewidth=1.2, linestyle="--", zorder=21)

        if not centers.empty:
            ax.scatter(
                centers["lon"].values,
                centers["lat"].values,
                s=26,
                marker="x",
                c="red",
                linewidths=1.1,
                zorder=22,
            )
            rp = centers.iloc[0]
            ax.text(
                rp["lon"] + 0.08,
                rp["lat"] + 0.05,
                test["group"],
                fontsize=8,
                color="red",
                zorder=23,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
            )
        elif len(boxes) > 0:
            rp = boxes.iloc[0].geometry.representative_point()
            ax.text(
                rp.x,
                rp.y,
                test["group"],
                fontsize=8,
                color="dodgerblue",
                zorder=23,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
            )

    ax.set_title("River/strait mask candidates overview", fontsize=12)

    fig.tight_layout()
    plotter.save(str(OUT_COMBINED), dpi=220)
    plt.close(fig)
    print(f"[OUTPUT] {OUT_COMBINED}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    mainland_gdf, coastline_gdf = build_mainland_and_coastline()

    dataset_dfs = []
    for spec in DATASETS:
        df = load_points_df(spec)
        if df is not None:
            dataset_dfs.append(df)

    if not dataset_dfs:
        print("[WARN] No datasets loaded.")
        return

    all_points = pd.concat(dataset_dfs, ignore_index=True)

    for test in CANDIDATE_TESTS:
        plot_single_group(
            group_name=test["group"],
            candidates=test["candidates"],
            zoom_pad_deg=float(test["zoom_pad_deg"]),
            map_zoom=int(test.get("map_zoom", MAP_ZOOM_DEFAULT)),
            all_points=all_points,
            mainland_gdf=mainland_gdf,
            coastline_gdf=coastline_gdf,
        )

    if SAVE_COMBINED_OVERVIEW:
        plot_combined_overview(CANDIDATE_TESTS)

    print("[DONE] River/strait mask tuning plots finished.")


if __name__ == "__main__":
    main()