#!/usr/bin/env python3
# Plot CTD station locations with temporal colorbar using MapPlotter

import MapPlotter as mp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path

# -----------------------------
# INPUT / OUTPUT
# -----------------------------
XLSX_FILE = Path("/Users/elena/Projects/IRB/MedClim/Data/Adriatic/CTD_data_HR_2019_2023_MFSD.xlsx")

# Change these to your local dirs when running on your Mac
OUTDIR = Path("/Users/elena/Projects/IRB/MedClim/Plots/Adriatic/CTD_XLSX")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# COLUMN DEFINITIONS (XLSX)
# -----------------------------
LON_COL   = "Z_SIRINA"
LAT_COL   = "Z_DUZINA"
TIME_COL  = "LOK_VRIJEME"

# -----------------------------
# READ + PARSE TIME
# Original format: "10:26 04.09.2021"
# -----------------------------
df = pd.read_excel(XLSX_FILE)

df["TIME"] = pd.to_datetime(
    df[TIME_COL],
    format="%H:%M %d.%m.%Y",
    errors="coerce",
)

# Drop rows with missing essentials
df = df.dropna(subset=[LON_COL, LAT_COL, "TIME"]).copy()

# MapPlotter expects timestamps if you use tick_format='datetime' in your setup
df["DATE"] = df["TIME"].astype("int64") // 10**9  # seconds since epoch (UTC-like)

print("Time range:", df["TIME"].min(), "→", df["TIME"].max())
print("N points:", len(df))

# -----------------------------
# MAP PLOTTER SETUP
# -----------------------------
plotter = mp.MapPlotter(projection="PlateCarree")
params = plotter.defaultParams()

params["dpi"] = 100

# Adriatic-ish bounds; tighten these if desired
params["xlim"] = [12.0, 20.5]
params["ylim"] = [40.5, 46.0]

params["features"]  = ["coastline"]
params["map_zoom"]  = 7
params["map_style"] = "satellite"

params["xlabel"] = ["Longitude (deg)"]
params["ylabel"] = ["Latitude (deg)"]

params["cmap"]        = "jet"
params["orientation"] = "vertical"
params["tick_format"] = "datetime"
params["shrink"]      = 0.6

fig = plt.figure(figsize=(12, 10))
ax  = fig.add_subplot(1, 1, 1, projection=plotter.projection)
params["fig"] = fig
params["ax"]  = ax

# -----------------------------
# DATA FOR PLOT
# -----------------------------
lon  = df[LON_COL].to_numpy()
lat  = df[LAT_COL].to_numpy()
time = df["DATE"].to_numpy()

# Plot time-colored scatter
plotter.scatter(lon, lat, data=time, size=10, params=params)

out_png = OUTDIR / "CTD_XLSX_Temporal_points.png"
plotter.save(str(out_png), dpi=300)

plt.clf()
plt.close()
print("Saved:", out_png)
