#!/usr/bin/env python3
# Reads : Data/Adriatic/CTD_data_HR_2019_2023_MFSD.xlsx
# Writes: Plots/Adriatic/CTD_HR_MFSD/CTD_HR_2019_2023_MFSD_long.csv and diagnostic figures
"""
Basic exploration of Adriatic CTD data from XLSX source.

- Read XLSX
- Parse datetime
- Print time / space / depth coverage
- Assign 3 Adriatic subregions
- Produce basic diagnostic plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import MapPlotter as mp

import Basins

# Repository-relative paths (Scripts/paths.py); MEDCLIM_DATA relocates Data/, Stats/, Plots/, Results/, Diagnostics/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Scripts.paths import DATA, PLOTS

# --------------------------------------------------
# INPUT
# --------------------------------------------------
XLSX_FILE = DATA / "Adriatic" / "CTD_data_HR_2019_2023_MFSD.xlsx"
OUTDIR = PLOTS / "Adriatic" / "CTD_HR_MFSD"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTDIR / "CTD_HR_2019_2023_MFSD_long.csv"

# --------------------------------------------------
# COLUMN DEFINITIONS
# --------------------------------------------------
LON_COL   = "Z_SIRINA"
LAT_COL   = "Z_DUZINA"
TIME_COL  = "LOK_VRIJEME"
DEPTH_COL = "DUBINA"
TEMP_COL  = "CTDTMP"
SAL_COL   = "CTDSAL"

# --------------------------------------------------
# READ DATA
# --------------------------------------------------
df = pd.read_excel(XLSX_FILE)

# --------------------------------------------------
# PARSE TIME
# Original format: "10:26 04.09.2021"
# --------------------------------------------------
df["time"] = pd.to_datetime(
    df[TIME_COL],
    format="%H:%M %d.%m.%Y",
    errors="coerce"
)

# --------------------------------------------------
# BASIC SANITY CHECKS
# --------------------------------------------------
print("\n--- BASIC INFO ---")
print(f"Rows: {len(df):,}")

print("\nTime range:")
print(df["time"].min(), "→", df["time"].max())

print("\nLatitude range:")
print(df[LAT_COL].min(), "→", df[LAT_COL].max())

print("\nLongitude range:")
print(df[LON_COL].min(), "→", df[LON_COL].max())

print("\nDepth range (m):")
print(df[DEPTH_COL].min(), "→", df[DEPTH_COL].max())

print("\nUnique stations (lon/lat):")
print(df[[LON_COL, LAT_COL]].drop_duplicates().shape[0])



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

df["BASIN_3ADR"] = basins_abbrev_for_lonlat(df[LON_COL].values, df[LAT_COL].values, basin_list)
print(df["BASIN_3ADR"].value_counts())

# --------------------------------------------------
# EXPORT UNIFORM "POINTS" CSV (same schema as netCDF exports)
# --------------------------------------------------

# Ensure numeric types
for col in [LON_COL, LAT_COL, DEPTH_COL, TEMP_COL, SAL_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows without essential coordinates/time/depth
df2 = df.dropna(subset=["time", LON_COL, LAT_COL, DEPTH_COL], how="any").copy()

# Build a station id from lon/lat (rounded to avoid float noise)
df2["STATION"] = (
    "MFSD_" +
    df2[LON_COL].round(5).astype(str) + "_" +
    df2[LAT_COL].round(5).astype(str)
)

df_out = pd.DataFrame({
    "TIME": df2["time"],
    "LONGITUDE": df2[LON_COL],
    "LATITUDE": df2[LAT_COL],
    "STATION": df2["STATION"],
    "BASIN_3ADR": df2["BASIN_3ADR"],
    "DEPTH_M": df2[DEPTH_COL],
    "TEMP_C": df2[TEMP_COL],
    "SAL_PSU": df2[SAL_COL],
})

# Optional: drop rows where both T and S missing
df_out = df_out.dropna(subset=["TEMP_C", "SAL_PSU"], how="all")
df_out["SOURCE"] = "HR_MFSD_XLSX"


df_out.to_csv(OUT_CSV, index=False)
print("\nSaved uniform CSV:", OUT_CSV)
print("Rows written:", len(df_out))
print("Unique stations:", df_out["STATION"].nunique())


# --------------------------------------------------
# BASIC PLOTS
# --------------------------------------------------

# 1) Map of stations
plt.figure(figsize=(6, 8))
plt.scatter(
    df[LON_COL],
    df[LAT_COL],
    c=df["BASIN_3ADR"].map({"NADR": "tab:blue", "CADR": "tab:green", "SADR": "tab:red"}),
    s=5,
    alpha=0.5
)
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.title("Adriatic CTD station locations")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) T and S vs depth (all data, cloud)
fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

ax[0].scatter(df[TEMP_COL], df[DEPTH_COL], s=2, alpha=0.3)
ax[0].set_xlabel("Temperature (°C)")
ax[0].set_ylabel("Depth (m)")
ax[0].invert_yaxis()
ax[0].grid(True)
ax[0].set_title("Temperature profiles")

ax[1].scatter(df[SAL_COL], df[DEPTH_COL], s=2, alpha=0.3)
ax[1].set_xlabel("Salinity (PSU)")
ax[1].invert_yaxis()
ax[1].grid(True)
ax[1].set_title("Salinity profiles")

plt.tight_layout()
plt.show()

# 3) Time coverage histogram
plt.figure(figsize=(8, 4))
df["time"].dt.year.value_counts().sort_index().plot(kind="bar")
plt.xlabel("Year")
plt.ylabel("Number of samples")
plt.title("Temporal distribution of samples")
plt.tight_layout()
plt.show()
