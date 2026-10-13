#!/usr/bin/env python3
# Reads : Data/Adriatic/IOR_Palagruza_Transect_1952_2018.nc
# Writes: Plots/Adriatic/Palagruza_Transect/Palagruza_Transect_1952_2018_long.csv and diagnostic figures
"""
Explore IOR Palagruza Transect netCDF (1952-2018):
- Print time/span, station list/locations, depths, missingness
- Add simple 3-region Adriatic division (lat-based)
- Plot station map with MapPlotter
- Plot basic time series (surface and selected depth) and a T-S diagram
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import MapPlotter as mp

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Scripts.paths import DATA, PLOTS


# -----------------------------
# INPUT / OUTPUT
# -----------------------------
NC_FILE = DATA / "Adriatic" / "IOR_Palagruza_Transect_1952_2018.nc"
OUTDIR  = PLOTS / "Adriatic" / "Palagruza_Transect"
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# SIMPLE 3-REGION ADRIATIC DIVISION (lat-based; replace later with polygons)
# -----------------------------

import Basins

north = Basins.adriatic.north
mid   = Basins.adriatic.mid
south = Basins.adriatic.south
basin_list = [north, mid, south]   # order matters if polygons overlap

def basins_abbrev_for_lonlat(lon, lat, basin_list, default="out"):
    """
    lon, lat: 1D numpy arrays (same length)
    returns: 1D object array of basin abbrev strings (e.g. 'nadr','cadr','sadr') or default
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    out = np.full(lon.shape, default, dtype=object)
    valid = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(valid):
        return out

    xyz = np.zeros((valid.sum(), 3), dtype=float)
    xyz[:, 0] = lon[valid]
    xyz[:, 1] = lat[valid]

    # fill in order; last match wins if overlap
    idx_valid = np.where(valid)[0]
    for basin in basin_list:
        mask = basin.areinside(xyz)
        out[idx_valid[mask]] = basin.abbrev

    return out


# -----------------------------
# READ DATASET
# -----------------------------
ds = xr.open_dataset(NC_FILE)

# Variables
T = ds["temperature"]  # (depth, station, time)
S = ds["salinity"]     # (depth, station, time)

time  = ds["time"].values
depth = ds["depth"].values
lon   = ds["station_lon"].values
lat   = ds["station_lat"].values

station_basin = basins_abbrev_for_lonlat(lon, lat, basin_list)

# Decode station names from (character, station) byte array
name_bytes = ds["station_name"].values  # shape (nchar, nstation)
station_names = []
for s in range(name_bytes.shape[1]):
    chars = name_bytes[:, s].tolist()
    nm = b"".join(chars).decode("utf-8", "ignore").strip("\x00").strip()
    station_names.append(nm)

# -----------------------------
# BASIC PRINTS
# -----------------------------
print("\n--- DATASET SUMMARY ---")
print(f"File: {NC_FILE}")
print(f"Stations: {len(station_names)}")
print(f"Depth levels: {len(depth)}")
print(f"Time steps: {len(time)}")

print("\nTime range:")
print(pd.to_datetime(time).min(), "→", pd.to_datetime(time).max())
# Confirm daily stepping
if len(time) > 1:
    dt_days = np.unique(np.diff(time).astype("timedelta64[D]").astype(int))
    print("Unique Δt (days):", dt_days[:10], "(showing up to 10 uniques)")

print("\nDepths (m):", depth)

print("\nStations (name, lon, lat):")
for nm, lo, la in zip(station_names, lon, lat):
    print(f"  {nm:>3s}  {lo:9.5f}  {la:9.5f} ")

# Missingness overview
T_valid = np.isfinite(T.values)
S_valid = np.isfinite(S.values)
print("\nValid fraction:")
print(f"  Temperature: {T_valid.mean():.3f}")
print(f"  Salinity:    {S_valid.mean():.3f}")

# -----------------------------
# EXPORT LONG (UNIFORM) FORMAT
# -----------------------------
df_long = ds[["temperature", "salinity"]].to_dataframe().reset_index()

df_long = df_long.rename(columns={
    "time": "TIME",
    "depth": "DEPTH_M",
    "temperature": "TEMP_C",
    "salinity": "SAL_PSU",
    "station": "STATION_IDX",
})

# station metadata (includes BASIN_3ADR from Basins polygons)
meta = pd.DataFrame({
    "STATION_IDX": np.arange(len(station_names), dtype=int),
    "STATION": station_names,
    "LONGITUDE": lon.astype(float),
    "LATITUDE": lat.astype(float),
    "BASIN_3ADR": station_basin,   # <-- from basins_abbrev_for_lonlat(lon,lat,...)
})

df_long = df_long.merge(meta, on="STATION_IDX", how="left")

df_long = df_long[[
    "TIME", "LONGITUDE", "LATITUDE", "STATION", "BASIN_3ADR",
    "DEPTH_M", "TEMP_C", "SAL_PSU"
]]

df_long = df_long.dropna(subset=["TEMP_C", "SAL_PSU"], how="all")

out_csv = OUTDIR / "Palagruza_Transect_1952_2018_long.csv"
df_long.to_csv(out_csv, index=False)
print("Saved long-format CSV:", out_csv)

print("\nStations per basin (polygon-based):")
print(pd.Series(station_basin).value_counts())


# -----------------------------
# MAP WITH MAPPLOTTER (stations only; fixed locations)
# -----------------------------
plotter = mp.MapPlotter(projection="PlateCarree")
params = plotter.defaultParams()

params["dpi"] = 100
params["xlim"] = [15.7, 16.6]    # tight around Palagruza transect; adjust if desired
params["ylim"] = [41.9, 43.5]
params["features"]  = ["coastline"]
params["map_zoom"]  = 8
params["map_style"] = "satellite"

params["xlabel"] = ["Longitude (deg)"]
params["ylabel"] = ["Latitude (deg)"]

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(1, 1, 1, projection=plotter.projection)
params["fig"] = fig
params["ax"]  = ax

# Color stations by index (temporal colormap is not meaningful for fixed stations)
station_idx = np.arange(len(station_names), dtype=float)

# Use scatter; data drives colorbar
params["draw_cbar"]        = False
#params["orientation"] = "vertical"
#params["shrink"]      = 0.7
#params["tick_format"] = "datetime"

plotter.scatter(lon, lat, data=station_idx, size=60, params=params)

# Add labels
for lo, la, nm in zip(lon, lat, station_names):
    ax.text(lo + 0.01, la + 0.01, nm, fontsize=10)

out_map = OUTDIR / "Palagruza_station_map.png"
plotter.save(str(out_map), dpi=300)
plt.clf()
plt.close()
print("\nSaved map:", out_map)


## -----------------------------
# BASIC TIME SERIES as SCATTER (all stations, multiple depths)
# -----------------------------
time_pd = pd.to_datetime(ds["time"].values)
depths_to_plot = list(depth)   # Palagruza has standard depths (e.g., 0..)

def scatter_multidepth(var_da, var_label, out_png):
    """
    Scatter all station values vs time for several depths.
    Uses xarray->dataframe, robust to dimension order.
    """
    dfv = var_da.to_dataframe(name="val").reset_index()
    dfv["time"] = pd.to_datetime(dfv["time"])
    dfv = dfv.dropna(subset=["val"])

    plt.figure(figsize=(12, 5))
    for z in depths_to_plot:
        dsub = dfv[dfv["depth"] == z]
        if dsub.empty:
            continue
        plt.scatter(dsub["time"], dsub["val"], s=5, alpha=0.20, label=f"{float(z):g} m")

    plt.xlabel("Time")
    plt.ylabel(var_label)
    plt.title(f"Palagruza Transect: {var_label} (all stations, scatter)")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved:", out_png)

scatter_multidepth(T, "Temperature (°C)", OUTDIR / "Temperature_scatter_multidepth.png")
scatter_multidepth(S, "Salinity (PSU)",    OUTDIR / "Salinity_scatter_multidepth.png")



# -----------------------------
# T–S DIAGRAM (multiple depths)
# -----------------------------
depths_to_plot = list(depth)  # or choose a subset

def ts_scatter_for_depth(z):
    daT = T.sel(depth=z)
    daS = S.sel(depth=z)
    TT = daT.values.reshape(-1)
    SS = daS.values.reshape(-1)
    m = np.isfinite(TT) & np.isfinite(SS)
    plt.scatter(SS[m], TT[m], s=2, alpha=0.15, label=f"{float(z):g} m")

plt.figure(figsize=(6, 6))
for z in depths_to_plot:
    ts_scatter_for_depth(z)

plt.xlabel("Salinity (PSU)")
plt.ylabel("Temperature (°C)")
plt.title("Palagruza Transect: T–S diagram (daily, all stations)")
plt.grid(True)
plt.legend(ncol=2, fontsize=9)
out_tsdiag = OUTDIR / "TS_diagram_multidepth.png"
plt.tight_layout()
plt.savefig(out_tsdiag, dpi=200)
plt.close()
print("Saved:", out_tsdiag)


# -----------------------------
# OPTIONAL: STATION-MEAN SCATTER (multiple depths)
# -----------------------------
MIN_STATIONS = 1

def scatter_station_mean(var_da, var_label, out_png):
    plt.figure(figsize=(12, 5))

    for z in depths_to_plot:
        da = var_da.sel(depth=z)
        n  = da.count(dim="station")                 # (time,)
        mu = da.mean(dim="station", skipna=True)     # (time,)
        mu = mu.where(n >= MIN_STATIONS)

        plt.scatter(pd.to_datetime(mu["time"].values), mu.values,
                    s=10, alpha=0.6, label=f"{float(z):g} m (n≥{MIN_STATIONS})")

    plt.xlabel("Time")
    plt.ylabel(var_label)
    plt.title(f"Palagruza Transect: {var_label} (station-mean, scatter)")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved:", out_png)

scatter_station_mean(T, "Temperature (°C)", OUTDIR / "Temperature_stationmean_scatter.png")
scatter_station_mean(S, "Salinity (PSU)",    OUTDIR / "Salinity_stationmean_scatter.png")

