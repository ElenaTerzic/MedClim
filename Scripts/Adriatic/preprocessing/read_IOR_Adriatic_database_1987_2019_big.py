#!/usr/bin/env python3
# Reads : Data/Adriatic/IOR_Adriatic_database_1987_2019.nc
# Writes: Plots/Adriatic/IOR_Adriatic_database_1987_2019/IOR_Adriatic_database_1987_2019_long.csv, IOR_database_station_map.png
"""
Huge netCDF streaming reader for:
IOR_Adriatic_database_1987_2019.nc

- Stage 1: fast inventory (time span, station map, cast counts) without reading profiles
- Stage 2: optional streaming export to long "points" CSV (same schema as the other sources)

Requires: netCDF4, numpy, pandas, matplotlib, MapPlotter
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MapPlotter as mp
from netCDF4 import Dataset, num2date

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Scripts.paths import DATA, PLOTS

# -----------------------------
# INPUT / OUTPUT
# -----------------------------
NC_FILE = DATA / "Adriatic" / "IOR_Adriatic_database_1987_2019.nc"
OUTDIR  = PLOTS / "Adriatic" / "IOR_Adriatic_database_1987_2019"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Export toggle (set False if you only want inventory/plots first)
DO_EXPORT = True

OUT_CSV = OUTDIR / "IOR_Adriatic_database_1987_2019_long.csv"

# Optional filters to keep export sane (recommended!)
# Bounding box (Adriatic)
USE_BBOX = True
LON_MIN, LON_MAX = 12.0, 20.5
LAT_MIN, LAT_MAX = 40.0, 46.0

# Time filter (set to None to disable)
TIME_MIN = pd.Timestamp("1987-01-01")
TIME_MAX = pd.Timestamp("2019-12-31")

# Limit stations/casts for testing
MAX_STATIONS = None
MAX_CASTS_PER_STATION = None

# -----------------------------
# 3-REGION ADRIATIC DIVISION 
# -----------------------------
import Basins

north = Basins.adriatic.north
mid   = Basins.adriatic.mid
south = Basins.adriatic.south
basin_list = [north, mid, south]  # order matters if overlap

def basins_abbrev_for_lonlat(lon, lat, basin_list, default="out"):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    out = np.full(lon.shape, default, dtype=object)

    valid = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(valid):
        return out

    xyz = np.zeros((valid.sum(), 3), dtype=float)
    xyz[:, 0] = lon[valid]
    xyz[:, 1] = lat[valid]

    idx_valid = np.where(valid)[0]
    for basin in basin_list:
        mask = basin.areinside(xyz)
        out[idx_valid[mask]] = basin.abbrev

    return out

# -----------------------------
# OPEN DATASET (netCDF4 streaming)
# -----------------------------
ds = Dataset(NC_FILE, mode="r")
ds.set_auto_mask(False)  # important: we handle _FillValue ourselves

lon = ds.variables["TS_lon"][:]  # (station,)
lat = ds.variables["TS_lat"][:]  # (station,)
tsec = ds.variables["TS_time"][:, :]  # (count, station) ~ 191x3265 -> manageable

station_basin = basins_abbrev_for_lonlat(lon, lat, basin_list)

print("Stations per basin (polygon):")
print(pd.Series(station_basin).value_counts())


fill_t = getattr(ds.variables["TS_time"], "_FillValue", -9999.9999)
valid_time_mask = np.isfinite(tsec) & (tsec != fill_t)

# Convert time range (only on valid)
time_units = ds.variables["TS_time"].units  # "seconds since 1979-01-01 00:00:00 UTC"
t_valid = tsec[valid_time_mask]

tmin = num2date(float(np.nanmin(t_valid)), units=time_units)
tmax = num2date(float(np.nanmax(t_valid)), units=time_units)

print("\n--- FAST INVENTORY ---")
print("File:", NC_FILE)
print("Stations:", lon.size)
print("Casts per station (TS_count):", ds.dimensions["TS_count"].size)
print("Max depth samples (TS_depth):", ds.dimensions["TS_depth"].size)
print("Time units:", time_units)

tmin = num2date(float(np.nanmin(t_valid)), units=time_units,
               only_use_cftime_datetimes=False, only_use_python_datetimes=True)
tmax = num2date(float(np.nanmax(t_valid)), units=time_units,
               only_use_cftime_datetimes=False, only_use_python_datetimes=True)

print("Time range (valid casts):", pd.Timestamp(tmin), "→", pd.Timestamp(tmax))


# Cast counts per station
n_casts_station = valid_time_mask.sum(axis=0)  # (station,)
print("Stations with >=1 cast:", int((n_casts_station > 0).sum()))
print("Total valid casts:", int(n_casts_station.sum()))

# Apply bbox station filter for reporting
if USE_BBOX:
    in_bbox = (lon >= LON_MIN) & (lon <= LON_MAX) & (lat >= LAT_MIN) & (lat <= LAT_MAX)
else:
    in_bbox = np.ones_like(lon, dtype=bool)

print("Stations in bbox:", int(in_bbox.sum()))

# -----------------------------
# MAP (stations only, no colorbar)
# -----------------------------
plotter = mp.MapPlotter(projection="PlateCarree")
params = plotter.defaultParams()
params["dpi"] = 100
params["features"] = ["coastline"]
params["map_zoom"] = 6
params["map_style"] = "satellite"
params["xlabel"] = ["Longitude (deg)"]
params["ylabel"] = ["Latitude (deg)"]
params["draw_cbar"] = False

# auto extents
padx, pady = 0.4, 0.4
params["xlim"] = [float(lon[in_bbox].min() - padx), float(lon[in_bbox].max() + padx)]
params["ylim"] = [float(lat[in_bbox].min() - pady), float(lat[in_bbox].max() + pady)]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=plotter.projection)
params["fig"] = fig
params["ax"] = ax

# Plot stations (no data -> no colorbar)
plotter.scatter(lon[in_bbox], lat[in_bbox], data=np.array([]), size=8, params=params)


out_map = OUTDIR / "IOR_database_station_map.png"
plotter.save(str(out_map), dpi=300)
plt.clf(); plt.close()
print("Saved map:", out_map)

# -----------------------------
# OPTIONAL: STREAMING EXPORT (profile-by-profile)
# -----------------------------
if DO_EXPORT:
    vD = ds.variables["TS_depth"]
    vT = ds.variables["TS_temperature"]
    vS = ds.variables["TS_salinity"]

    fill_D = getattr(vD, "_FillValue", -9999.9999)
    fill_T = getattr(vT, "_FillValue", -9999.9999)
    fill_S = getattr(vS, "_FillValue", -9999.9999)

    # Prepare output
    if OUT_CSV.exists():
        OUT_CSV.unlink()  # avoid accidental append to old file

    header_written = False

    station_indices = np.where(in_bbox)[0]
    if MAX_STATIONS is not None:
        station_indices = station_indices[:MAX_STATIONS]

    print("\n--- STREAMING EXPORT ---")
    print("Writing:", OUT_CSV)
    print("Stations to process:", len(station_indices))

    # Track min/max depth encountered (true data)
    global_depth_min = np.inf
    global_depth_max = -np.inf

    for ii, i in enumerate(station_indices):
        lo = float(lon[i])
        la = float(lat[i])
        bas = station_basin[i]
        station_name = f"IOR_{i:04d}"  # you can change this

        # which casts at this station have valid time?
        valid_casts = np.where(valid_time_mask[:, i])[0]
        if MAX_CASTS_PER_STATION is not None:
            valid_casts = valid_casts[:MAX_CASTS_PER_STATION]

        # Optional time filter (cheap: only filter by TS_time)
        if TIME_MIN is not None or TIME_MAX is not None:
            # seconds since 1979-01-01
            origin = pd.Timestamp("1979-01-01 00:00:00")
            tcast = tsec[valid_casts, i].astype(float)

            keep = np.ones(len(valid_casts), dtype=bool)

            if TIME_MIN is not None:
                tmin_sec = (pd.Timestamp(TIME_MIN) - origin).total_seconds()
                keep &= (tcast >= tmin_sec)

            if TIME_MAX is not None:
                tmax_sec = (pd.Timestamp(TIME_MAX) - origin).total_seconds()
                keep &= (tcast <= tmax_sec)

            valid_casts = valid_casts[keep]


        if valid_casts.size == 0:
            continue

        for j in valid_casts:
            # Cast time
            tt = float(tsec[j, i])
            if not np.isfinite(tt) or tt == fill_t:
                continue
            dt = num2date(tt, units=time_units,
              only_use_cftime_datetimes=False, only_use_python_datetimes=True)
            origin = pd.Timestamp("1979-01-01 00:00:00")
            tdt = origin + pd.to_timedelta(tt, unit="s")


            # Read ONE profile (depth axis only) -> arrays length TS_depth
            D = vD[:, j, i].astype(float)
            TT = vT[:, j, i].astype(float)
            SS = vS[:, j, i].astype(float)

            # Mask fill values
            mD = np.isfinite(D) & (D != fill_D)
            mT = np.isfinite(TT) & (TT != fill_T)
            mS = np.isfinite(SS) & (SS != fill_S)

            # Keep rows where at least depth exists AND (T or S exists)
            m = mD & (mT | mS)
            if not np.any(m):
                continue

            Dm = D[m]
            Tm = np.where(mT[m], TT[m], np.nan)
            Sm = np.where(mS[m], SS[m], np.nan)

            global_depth_min = min(global_depth_min, float(np.nanmin(Dm)))
            global_depth_max = max(global_depth_max, float(np.nanmax(Dm)))

            # Build chunk dataframe
            df_chunk = pd.DataFrame({
                "TIME": tdt,
                "LONGITUDE": lo,
                "LATITUDE": la,
                "STATION": station_name,
                "CAST_IDX": int(j),
                "BASIN_3ADR": bas,
                "DEPTH_M": Dm,
                "TEMP_C": Tm,
                "SAL_PSU": Sm,
            })


            # Append to CSV
            df_chunk.to_csv(
                str(OUT_CSV),
                mode="a",
                index=False,
                header=(not header_written),
            )
            header_written = True

        if (ii + 1) % 50 == 0:
            print(f"  processed {ii+1}/{len(station_indices)} stations ...")

    print("Export done.")
    if np.isfinite(global_depth_min):
        print(f"Depth range encountered (m): {global_depth_min:.2f} → {global_depth_max:.2f}")

ds.close()
print("\nDone.")
