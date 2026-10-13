#!/usr/bin/env python3
# Reads : Data/Adriatic/RB_Northern_Adriatic_Transect_1979_2018.nc
# Writes: Plots/Adriatic/Northern_Adriatic_Transect/RB_Northern_Adriatic_Transect_1979_2018_long.csv and diagnostic figures
"""
RB Northern Adriatic Transect netCDF (1979-2018):
- Print time span, station list/locations, depths, missingness
- MapPlotter: station map with station-name colorbar (NOT datetime)
- Basic time series (station-mean at surface and 20 m)
- Basic T–S diagram (surface + 20 m)
- Export long-format CSV in the same "points" schema
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
NC_FILE = DATA / "Adriatic" / "RB_Northern_Adriatic_Transect_1979_2018.nc"
OUTDIR  = PLOTS / "Adriatic" / "Northern_Adriatic_Transect"
OUTDIR.mkdir(parents=True, exist_ok=True)


import Basins

north = Basins.adriatic.north
mid   = Basins.adriatic.mid
south = Basins.adriatic.south
basin_list = [north, mid, south]   # order matters if polygons overlap

# -----------------------------
# ASSIGN BASINS
# -----------------------------

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
# READ DATA
# -----------------------------
ds = xr.open_dataset(NC_FILE)

T = ds["temperature"]  # (depth, station, time)
S = ds["salinity"]     # (depth, station, time)

time  = pd.to_datetime(ds["time"].values)
depth = ds["depth"].values.astype(float)
lon   = ds["station_lon"].values.astype(float)
lat   = ds["station_lat"].values.astype(float)

station_basin = basins_abbrev_for_lonlat(lon, lat, basin_list)

# Decode station names from (character, station) byte array
name_bytes = ds["station_name"].values  # (nchar, nstation)
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
print(f"Depth levels: {len(depth)}  -> {depth}")
print(f"Time steps: {len(time)}")
print("Time range:", time.min(), "→", time.max())

if len(time) > 1:
    dt_days = np.unique(np.diff(time.values).astype("timedelta64[D]").astype(int))
    print("Unique Δt (days):", dt_days[:10], "(showing up to 10 uniques)")

print("\nStations (name, lon, lat):")
for nm, lo, la in zip(station_names, lon, lat):
    print(f"  {nm:>5s}  {lo:10.5f}  {la:10.5f}")

T_valid = np.isfinite(T.values).mean()
S_valid = np.isfinite(S.values).mean()
print("\nValid fraction:")
print(f"  Temperature: {T_valid:.3f}")
print(f"  Salinity:    {S_valid:.3f}")

# -----------------------------
# MAP (stations only) with MapPlotter + station-name colorbar
# -----------------------------
plotter = mp.MapPlotter(projection="PlateCarree")
params = plotter.defaultParams()

params["dpi"] = 100
# auto bounds (small padding)
padx, pady = 0.15, 0.15
params["xlim"] = [float(lon.min()-padx), float(lon.max()+padx)]
params["ylim"] = [float(lat.min()-pady), float(lat.max()+pady)]

params["features"]  = ["coastline"]
params["map_zoom"]  = 8
params["map_style"] = "satellite"
params["xlabel"] = ["Longitude (deg)"]
params["ylabel"] = ["Latitude (deg)"]

params["draw_cbar"]   = False
#params["orientation"] = "vertical"
#params["shrink"]      = 0.7

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(1, 1, 1, projection=plotter.projection)
params["fig"] = fig
params["ax"]  = ax

station_idx = np.arange(len(station_names), dtype=float)
plotter.scatter(lon, lat, data=station_idx, size=70, params=params)

# Label stations
for lo, la, nm in zip(lon, lat, station_names):
    ax.text(lo + 0.01, la + 0.01, nm, fontsize=10)

# Build our own colorbar with station labels
#mappable = ax.collections[-1]
#cb = plt.colorbar(mappable, ax=ax, orientation="vertical", shrink=0.7)
#cb.set_ticks(station_idx)
#cb.set_ticklabels(station_names)
#cb.set_label("Station")

out_map = OUTDIR / "RB_NAdriatic_station_map.png"
plotter.save(str(out_map), dpi=300)
plt.clf()
plt.close()
print("\nSaved map:", out_map)

# -----------------------------
# BASIC TIME SERIES (station-mean for multiple depths)
# -----------------------------

# Choose which depths to show:
# Option A: all depths in the file
depths_to_plot = list(depth)

# Option B: only these target depths (nearest available)
# targets = [0, 5, 10, 20, 30, 40]
# depths_to_plot = [depth[np.argmin(np.abs(depth - z))] for z in targets]
# depths_to_plot = sorted(set(depths_to_plot))

MIN_STATIONS = 1  # set to 2 or 3 if you want stricter coverage

# -----------------------------
# BASIC TIME SERIES as SCATTER (all stations, multiple depths)
# -----------------------------

depths_to_plot = list(depth)   # [0, 5, 10, 20, 30, 40] in this dataset

def scatter_multidepth(var_da, var_label, out_png):
    """
    Scatter all station values vs time for several depths.
    Robust to dimension order because we use xarray -> dataframe.
    """
    # long table with columns: depth, station, time, value
    dfv = var_da.to_dataframe(name="val").reset_index()
    dfv["time"] = pd.to_datetime(dfv["time"])
    dfv = dfv.dropna(subset=["val"])

    plt.figure(figsize=(12, 5))

    for z in depths_to_plot:
        # pick exact depth match (these files have exact standard depths)
        dsub = dfv[dfv["depth"] == z]
        if dsub.empty:
            # fallback: nearest depth if needed
            z_near = float(dfv["depth"].iloc[(dfv["depth"] - z).abs().argsort()[:1]].values[0])
            dsub = dfv[dfv["depth"] == z_near]
            z = z_near

        plt.scatter(dsub["time"], dsub["val"], s=6, alpha=0.25, label=f"{z:g} m")

    plt.xlabel("Time")
    plt.ylabel(var_label)
    plt.title(f"RB Northern Adriatic Transect: {var_label} (all stations, scatter)")
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
plt.title("RB Northern Adriatic Transect: T–S diagram (daily, all stations)")
plt.grid(True)
plt.legend(ncol=2, fontsize=9)
out_tsdiag = OUTDIR / "TS_diagram_multidepth.png"
plt.tight_layout()
plt.savefig(out_tsdiag, dpi=200)
plt.close()
print("Saved:", out_tsdiag)

# -----------------------------
# STATION-MEAN SCATTER (robust)
# -----------------------------
MIN_STATIONS = 1

# -----------------------------
# OPTIONAL: STATION-MEAN SCATTER
# -----------------------------
MIN_STATIONS = 1

def scatter_station_mean(var_da, var_label, out_png):
    plt.figure(figsize=(12, 5))
    for z in depths_to_plot:
        da = var_da.sel(depth=z)
        n  = da.count(dim="station")
        mu = da.mean(dim="station", skipna=True).where(n >= MIN_STATIONS)
        plt.scatter(pd.to_datetime(mu["time"].values), mu.values, s=10, alpha=0.6,
                    label=f"{float(z):g} m (n≥{MIN_STATIONS})")
    plt.xlabel("Time")
    plt.ylabel(var_label)
    plt.title(f"RB Northern Adriatic Transect: {var_label} (station-mean, scatter)")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved:", out_png)

scatter_station_mean(T, "Temperature (°C)", OUTDIR / "Temperature_stationmean_scatter.png")
scatter_station_mean(S, "Salinity (PSU)",    OUTDIR / "Salinity_stationmean_scatter.png")

# -----------------------------
# EXPORT LONG CSV (uniform "points" schema) + Basins polygons
# -----------------------------
df_long = ds[["temperature", "salinity"]].to_dataframe().reset_index()

df_long = df_long.rename(columns={
    "time": "TIME",
    "depth": "DEPTH_M",
    "temperature": "TEMP_C",
    "salinity": "SAL_PSU",
    "station": "STATION_IDX",
})

meta = pd.DataFrame({
    "STATION_IDX": np.arange(len(station_names), dtype=int),
    "STATION": station_names,
    "LONGITUDE": lon,
    "LATITUDE": lat,
    "BASIN_3ADR": station_basin,   # <-- from basins_abbrev_for_lonlat(lon,lat,...)
})

df_long = df_long.merge(meta, on="STATION_IDX", how="left")

df_long = df_long[[
    "TIME", "LONGITUDE", "LATITUDE", "STATION", "BASIN_3ADR",
    "DEPTH_M", "TEMP_C", "SAL_PSU"
]].dropna(subset=["TEMP_C", "SAL_PSU"], how="all")

out_csv = OUTDIR / "RB_Northern_Adriatic_Transect_1979_2018_long.csv"
df_long.to_csv(str(out_csv), index=False)
print("Saved long-format CSV:", out_csv)



# -----------------------------
# DATA AVAILABILITY (n valid stations) as a quick diagnostic
# -----------------------------
plt.figure(figsize=(12, 5))

for z in depths_to_plot:
    iz = int(np.argmin(np.abs(depth - z)))
    V = T.isel(depth=iz).values
    n_valid = np.isfinite(V).sum(axis=0)
    plt.plot(time, n_valid, label=f"{depth[iz]:g} m")

plt.xlabel("Time")
plt.ylabel("Valid stations (count)")
plt.title("Data availability by depth (Temperature)")
plt.grid(True)
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
out_av = OUTDIR / "Availability_by_depth_T.png"
plt.savefig(out_av, dpi=200)
plt.close()
print("Saved:", out_av)

