import pandas as pd
import numpy as np

# ---- column names (adjust ONLY if needed) ----
DT_COL = "yyyy-mm-ddThh:mm:ss.sss"
LON_COL = "Longitude [degrees East]"
LAT_COL = "Latitude [degrees North]"
P_COL   = "Pressure [dbar]"

# Read file --------------------------------------------
df = pd.read_csv("/Users/elena/Projects/IRB/Manca_Argo/Data/Medatlas/CSV/data_from_MedatlasII_allTS.csv")  # adjust sep if needed


# Normalize datetime column
s = df[DT_COL].astype(str).str.strip()
s = s.replace({"": np.nan, "nan": np.nan})

# Detect date precision
has_year  = s.str.match(r"^\d{4}$")
has_ym    = s.str.match(r"^\d{4}-\d{2}$")
has_ymd   = s.str.match(r"^\d{4}-\d{2}-\d{2}$")

# Parse datetime (year-only becomes Jan 1 by pandas — we WILL NOT trust it)
dt = pd.to_datetime(s, errors="coerce", utc=True)

df["datetime"]   = dt
df["has_year"]   = has_year
df["has_month"]  = has_ym | has_ymd
df["has_day"]    = has_ymd

# Build profile ID (same logic as NetCDF dedup)
df["lon_r"] = df[LON_COL].round(2)
df["lat_r"] = df[LAT_COL].round(2)

# Use date *string* for profile ID, not parsed datetime
df["date_str"] = s

profile_id = (
    df["lon_r"].astype(str) + "|" +
    df["lat_r"].astype(str) + "|" +
    df["date_str"]
)

df["profile_id"] = profile_id

# Reduce to unique profiles
profiles = (
    df.groupby("profile_id")
      .agg(
          lon=("lon_r", "first"),
          lat=("lat_r", "first"),
          date_str=("date_str", "first"),
          has_month=("has_month", "any"),
          has_day=("has_day", "any"),
          year=("datetime", lambda x: x.dt.year.iloc[0] if x.notna().any() else np.nan),
          month=("datetime", lambda x: x.dt.month.iloc[0] if x.notna().any() else np.nan),
      )
      .reset_index(drop=True)
)

# ---------------------------------------------
# GLOBAL COUNTS
# ---------------------------------------------
n_total = len(profiles)
n_year_only = (~profiles["has_month"]).sum()
n_with_month = profiles["has_month"].sum()
n_with_day = profiles["has_day"].sum()

print("\n=== GLOBAL PROFILE COUNTS ===")
print(f"Total profiles:                 {n_total:,}")
print(f"Profiles with YEAR ONLY:        {n_year_only:,}  ({n_year_only/n_total:.1%})")
print(f"Profiles with MONTH info:       {n_with_month:,}  ({n_with_month/n_total:.1%})")
print(f"Profiles with DAY info:         {n_with_day:,}  ({n_with_day/n_total:.1%})")

# ---------------------------------------------
# 1950–1999 SUBSET
# ---------------------------------------------
p50 = profiles[(profiles["year"] >= 1950) & (profiles["year"] <= 1999)]

print("\n=== 1950–1999 ONLY ===")
print(f"Profiles in period:             {len(p50):,}")
print(f"YEAR ONLY:                      {(~p50['has_month']).sum():,}")
print(f"WITH MONTH:                     {p50['has_month'].sum():,}")
print(f"WITH DAY:                       {p50['has_day'].sum():,}")

# ---------------------------------------------
# FAKE JANUARY DIAGNOSTIC
# ---------------------------------------------
fake_jan = p50[(p50["month"] == 1) & (~p50["has_month"])]
real_jan = p50[(p50["month"] == 1) & (p50["has_month"])]

print("\n=== JANUARY DIAGNOSTIC (1950–1999) ===")
print(f"January profiles TOTAL:         {(p50['month']==1).sum():,}")
print(f"  of which REAL January:        {len(real_jan):,}")
print(f"  of which FAKE January:        {len(fake_jan):,}")
print(f"Fraction of January that is fake: {len(fake_jan)/max(1,(p50['month']==1).sum()):.1%}")
