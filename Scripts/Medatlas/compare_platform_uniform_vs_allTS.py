import os
import re
import numpy as np
import pandas as pd
import xarray as xr

import Basins


# ---------------- SETTINGS ----------------
DATA_DIR = "/Users/elena/Projects/IRB/Manca_Argo/Data/Medatlas/NetCDF"
FILES = {
    "Bottle": os.path.join(DATA_DIR, "Btl_data.nc"),
    "CTD":    os.path.join(DATA_DIR, "CTD_data.nc"),
    "XBT":    os.path.join(DATA_DIR, "XBT_data.nc"),
    "MBT":    os.path.join(DATA_DIR, "MBT_data.nc"),
    "allTS":  os.path.join(DATA_DIR, "allTS_data.nc"),
}

basin = Basins.worldseas.med
YEAR_MIN = 1908
YEAR_MAX = 1999

# key rounding (match what you used before)
ROUND_LONLAT = 2
ROUND_DAYS   = 3
# -----------------------------------------


def decode_odv_string(var):
    a = np.asarray(var.values if hasattr(var, "values") else var)

    if a.ndim == 1 and a.dtype.kind in ("S", "U"):
        out = []
        for x in a:
            if isinstance(x, (bytes, np.bytes_)):
                out.append(x.decode("utf-8", errors="ignore").strip())
            else:
                out.append(str(x).strip())
        return np.array(out, dtype=object)

    if a.ndim == 2 and a.dtype.kind == "U":
        return np.array(["".join(row).strip() for row in a], dtype=object)

    if a.ndim == 2 and a.dtype.kind == "S":
        out = []
        for row in a:
            out.append(b"".join(row).decode("utf-8", errors="ignore").strip())
        return np.array(out, dtype=object)

    if a.ndim == 2 and a.dtype.kind in ("i", "u"):
        out = []
        for row in a:
            row = np.asarray(row)
            if (row == 0).any():
                row = row[:np.argmax(row == 0)]
            out.append(bytes(row.tolist()).decode("utf-8", errors="ignore").strip())
        return np.array(out, dtype=object)

    return a.astype(str)


def basin_mask_station(lon, lat, basin):
    xyz = np.zeros((len(lon), 3), dtype=float)
    xyz[:, 0] = lon
    xyz[:, 1] = lat
    good = np.isfinite(lon) & np.isfinite(lat)
    m = np.zeros(len(lon), dtype=bool)
    m[good] = basin.areinside(xyz[good])
    return m


def years_days_dt_from_date_time(ds):
    days = ds["date_time"].values.astype(float)
    fill = ds["date_time"].attrs.get("_FillValue", None)

    bad = ~np.isfinite(days)
    if fill is not None:
        bad |= (days == fill)

    days = days.copy()
    days[bad] = np.nan

    units = ds["date_time"].attrs.get("units", "")
    m = re.search(r"days since (\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", units)
    if not m:
        raise ValueError(f"Cannot parse date_time units: {units}")

    date0, time0 = m.group(1), m.group(2)
    t0 = pd.Timestamp(f"{date0}T{time0}Z")

    # absolute datetimes
    dt = (t0 + pd.to_timedelta(days, unit="D")).to_numpy(dtype="datetime64[s]")

    years = pd.DatetimeIndex(dt).year.to_numpy(dtype=float)
    years = np.where(np.isfinite(days), years, np.nan)

    return years, days, dt



def station_has_any_valid(ds, varname, station_mask):
    if varname not in ds.variables:
        return np.zeros(len(station_mask), dtype=bool)

    data = ds[varname].values
    fill = ds[varname].attrs.get("_FillValue", None)

    valid = np.isfinite(data)
    if fill is not None:
        valid &= (data != fill)

    has_any = valid.any(axis=1)
    return station_mask & has_any


def build_key_df(nc_path, platform_label):
    ds = xr.open_dataset(nc_path, decode_times=False)

    lon = ds["longitude"].values.astype(float)
    lat = ds["latitude"].values.astype(float)

    m_basin = basin_mask_station(lon, lat, basin)

    years, days, dt = years_days_dt_from_date_time(ds)
    tkey = dt.astype("datetime64[s]").astype("int64")  # absolute seconds since epoch

    valid_year = np.isfinite(years) & (years >= YEAR_MIN) & (years <= YEAR_MAX)

    # Require at least 1 valid Temperature sample (Fig.3-like station definition)
    base_mask = m_basin & valid_year
    cast_mask = station_has_any_valid(ds, "Temperature", base_mask)

    cruise = decode_odv_string(ds["cruise_id"]) if "cruise_id" in ds.variables else None
    station = decode_odv_string(ds["station_id"]) if "station_id" in ds.variables else None

    # fallbacks if cruise/station missing
    if cruise is None:
        cruise = np.array([""] * len(lon), dtype=object)
    if station is None:
        station = np.array([""] * len(lon), dtype=object)

    lon_r = np.round(lon, ROUND_LONLAT)
    lat_r = np.round(lat, ROUND_LONLAT)
    days_r = np.round(days, ROUND_DAYS)

    tday = dt.astype("datetime64[D]").astype("int64")  # day bins

    df = pd.DataFrame({
        "platform": platform_label,
        "lon": np.round(lon, ROUND_LONLAT)[cast_mask],
        "lat": np.round(lat, ROUND_LONLAT)[cast_mask],
        "tday": tday[cast_mask],
        "year": years[cast_mask].astype(int),
    })

    df["key"] = (
        df["lon"].astype(str) + "|" +
        df["lat"].astype(str) + "|" +
        df["tday"].astype(str)
    )



    return df


def main():
    # ---- build per-platform dataframes ----
    dfs = {}
    for plat in ["Bottle", "CTD", "XBT", "MBT"]:
        path = FILES[plat]
        print(f"Reading {plat}: {path}")
        dfs[plat] = build_key_df(path, plat)

    df_sum = pd.concat([dfs["Bottle"], dfs["CTD"], dfs["XBT"], dfs["MBT"]], ignore_index=True)

    # S = platform-sum records; U = unique union stations across platforms
    S = len(df_sum)
    df_union = df_sum.drop_duplicates(subset=["key"])
    U = len(df_union)
    O = S - U

    print("\n================ PLATFORM UNION DIAGNOSTIC ================")
    print(f"Platform sum S (Bottle+CTD+XBT+MBT): {S:,}")
    print(f"Unique union U (dedup across platforms): {U:,}")
    print(f"Overlaps O = S - U (double-counts in platform sum): {O:,}")

    # overlaps by year (where differences happen)
    per_year = df_sum.groupby("year").size().rename("S_platform_sum")
    per_year_U = df_union.groupby("year").size().rename("U_platform_union")
    per_year_O = (per_year - per_year_U).rename("O_overlaps")

    yearly_diag = pd.concat([per_year, per_year_U, per_year_O], axis=1).fillna(0).astype(int)
    yearly_diag = yearly_diag.loc[(yearly_diag.index >= YEAR_MIN) & (yearly_diag.index <= YEAR_MAX)]

    print("\nTop years by overlaps (platform double-counting):")
    print(yearly_diag.sort_values("O_overlaps", ascending=False).head(15).to_string())

    # ---- compare against allTS ----
    print("\n================ allTS COMPARISON ================")
    print(f"Reading allTS: {FILES['allTS']}")
    df_all = build_key_df(FILES["allTS"], "allTS")

    all_raw = len(df_all)
    all_unique = len(df_all.drop_duplicates(subset=["key"]))

    print(f"allTS raw (after mask+year+Temperature filter): {all_raw:,}")
    print(f"allTS unique (dedup within allTS):             {all_unique:,}")
    print(f"Difference (U_platform_union - allTS_unique):  {U - all_unique:,}")

    # where the union differs from allTS (diagnostic)
    keys_union = set(df_union["key"].values)
    keys_all = set(df_all.drop_duplicates(subset=["key"])["key"].values)

    only_in_union = len(keys_union - keys_all)
    only_in_all = len(keys_all - keys_union)

    print(f"Keys only in platform-union (not in allTS): {only_in_union:,}")
    print(f"Keys only in allTS (not in platform-union): {only_in_all:,}")

    print("\n(If those last two numbers are small-ish vs totals, you're in great shape;")
    print("they usually reflect export composition or tiny metadata differences.)")


if __name__ == "__main__":
    main()
