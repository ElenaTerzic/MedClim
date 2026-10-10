#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Identify the SeaDataNet casts that are Argo float profiles already held by the Argo set.

Why
---
The SeaDataNet aggregation (SDC_MED_DATA_TS_V2) archives profiling-float cycles as
ordinary casts, under several cruise names: "PF_<WMO>" (station "<WMO>_<cycle>_<A|D>"),
"Argo <WMO>" (station "<cycle>_<mode>_<A|D>") and "ARGO-ITALY ..." (running station
numbers); a national database (cruise "SHODB") holds some more as "CTD".  Most of them
are also in the Argo set used here, which holds ascending profiles only.  The pipeline
removes duplicates within a source only, so without this step these profiles enter every
pooled product twice (21,926 casts of 2005-2019 in the Mediterranean master file; found
during the Ocean Science revision, 5 Oct 2026).  This script lists them, and
remove_argo_copies_from_seadatanet.py writes the SeaDataNet master file without them.

A SeaDataNet cast is a copy when the Argo set holds the same profile:

  id             the entry names its float, cycle and direction, it is an ascending
                 profile, and the Argo set holds that float and cycle within ID_MAX_HOURS;
  time-position  the entry is named as a float and an Argo-set profile lies within
                 TP_HOURS and TP_DEG of it;
  content        any SeaDataNet cast with an Argo-set profile within one day and
                 CONTENT_DEG whose binned temperatures are the same (at least
                 CONTENT_MIN_BINS common bins, one of them at 100 m or above, largest
                 difference CONTENT_MAX_DT).

Descending float profiles and float profiles the Argo set does not hold are not copies
and stay in the data.  For every copy the list records the twin and how well the two
agree, so the matching can be checked.

Inputs   the SeaDataNet and Argo master files after basin assignment and the coastal mask
Outputs  Data/Seadatanet/sdn_argo_copies.csv            full list (copies and other
                                                        float-named entries, with twins)
         Scripts/Seadatanet/preprocessing/sdn_argo_copies_pids.txt
                                                        the copies' cast ids, one per
                                                        line (tracked in the repository)

The cast id is the pipeline's: "SDN|<LOCAL_CDI_ID>|<Cruise>|<Station>" for SeaDataNet,
"ARGO|<PLATFORM_NUMBER>|<CYCLE_NUMBER>" for Argo.  Streams both master files once
(about 10 min on a desktop, 30-40 min on a laptop); the logic is the one used for the
Ocean Science revision (os_revision_float_copies.py) and reproduces its list exactly.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SDN = PROJECT_ROOT / "Data/Seadatanet/Seadatanet.profiles_only_with_basins_mainland_special_excluded.csv"
ARGO = PROJECT_ROOT / "Data/Argo/CSV/Argo_with_basins_mainland_special_excluded.csv"
OUT_LIST = PROJECT_ROOT / "Data/Seadatanet/sdn_argo_copies.csv"
OUT_PIDS = Path(__file__).resolve().parent / "sdn_argo_copies_pids.txt"

Y0, Y1 = 1950, 2025
CHUNK = 2_000_000
GOOD = ["1", "2"]

# bins of the per-cast temperature table used by the "content" rule
CENTERS = np.array([5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300, 400,
                    500, 600, 700, 800, 900, 1000, 1100, 1200], float)
EDGES = np.concatenate([[0.0], 0.5 * (CENTERS[:-1] + CENTERS[1:]),
                        [CENTERS[-1] + 0.5 * (CENTERS[-1] - CENTERS[-2])]])
BINS = [int(c) for c in CENTERS]
TCOLS = [f"T{z}" for z in BINS]
SCOLS = [f"S{z}" for z in BINS]

FLOAT_CRUISE = r"^(?:PF_|ARGO-ITALY|Argo \d)"
ID_MAX_HOURS = 24.0
TP_HOURS, TP_DEG = 3.0, 0.05
CONTENT_DEG, CONTENT_MIN_BINS, CONTENT_MAX_DT = 0.1, 5, 0.01


# ==========================================================
# PER-CAST TABLES
# ==========================================================

def _accumulate(d: pd.DataFrame, col: str) -> pd.DataFrame:
    bi = np.searchsorted(EDGES, d.z.values, side="right") - 1
    ok = (bi >= 0) & (bi < len(CENTERS))
    d = d[ok].assign(bin=CENTERS[bi[ok]].astype(int))
    return d.groupby(["pid", "bin"], sort=False)[col].agg(["sum", "count"])


def _finish(sums: dict, metas: list, source: str) -> pd.DataFrame:
    meta = pd.concat(metas).reset_index()
    agg = {k: (k, "first") for k in ("time", "lon", "lat", "basin", "instr")}
    agg["zmax"] = ("zmax", "max")
    meta = meta.groupby("pid", sort=False).agg(**agg)
    for var, parts in sums.items():
        s = pd.concat(parts).groupby(level=[0, 1]).sum()
        tab = (s["sum"] / s["count"]).unstack("bin").reindex(columns=BINS)
        tab.columns = [f"{var}{z}" for z in BINS]
        meta = meta.join(tab, how="left")
    out = meta.reset_index()
    out["source"] = source
    return out


def build_sdn() -> pd.DataFrame:
    cols = {"Cruise": "cruise", "Station:METAVAR:INDEXED_TEXT": "station",
            "yyyy-mm-ddThh:mm:ss.sss": "time", "Longitude [degrees_east]": "lon",
            "Latitude [degrees_north]": "lat", "LOCAL_CDI_ID": "cdi", "Depth [m]": "z",
            "ITS-90 water temperature [degrees C]": "T", "QV:SEADATANET.1": "qT",
            "Water body salinity [per mille]": "S", "QV:SEADATANET.2": "qS",
            "Instrument / gear type:METAVAR:INDEXED_TEXT": "instr", "basin": "basin"}
    sums, metas, n, t0 = {"T": [], "S": []}, [], 0, time.time()
    for ch in pd.read_csv(SDN, usecols=list(cols), dtype=str, chunksize=CHUNK):
        n += len(ch)
        ch = ch.rename(columns=cols)
        yr = ch.time.str[:4]
        ch = ch[(yr >= str(Y0)) & (yr <= str(Y1))]
        if not len(ch):
            continue
        for k in ("lon", "lat", "z", "T", "S"):
            ch[k] = pd.to_numeric(ch[k], errors="coerce")
        ch["pid"] = "SDN|" + ch.cdi.fillna("") + "|" + ch.cruise.fillna("") + "|" + ch.station.fillna("")
        ch = ch[np.isfinite(ch.z) & (ch.z >= 0) & (ch.z <= 6000)]
        metas.append(ch.groupby("pid", sort=False).agg(
            time=("time", "first"), lon=("lon", "first"), lat=("lat", "first"),
            basin=("basin", "first"), instr=("instr", "first"), zmax=("z", "max")))
        sums["T"].append(_accumulate(ch[ch.qT.str.strip().isin(GOOD) & np.isfinite(ch["T"])
                                        & (ch["T"] >= -5) & (ch["T"] <= 45)], "T"))
        sums["S"].append(_accumulate(ch[ch.qS.str.strip().isin(GOOD) & np.isfinite(ch.S)
                                        & (ch.S >= 0) & (ch.S <= 60)], "S"))
        print(f"    SeaDataNet: {n/1e6:.0f}M rows, {time.time()-t0:.0f} s", flush=True)
    return _finish(sums, metas, "SDN")


def build_argo() -> pd.DataFrame:
    use = ["CYCLE_NUMBER", "PLATFORM_NUMBER", "PRES", "PRES_QC", "TEMP", "TEMP_QC",
           "PSAL", "PSAL_QC", "LATITUDE", "LONGITUDE", "TIME", "basin"]
    sums, metas, n, t0 = {"T": [], "S": []}, [], 0, time.time()
    for ch in pd.read_csv(ARGO, usecols=use, dtype=str, chunksize=CHUNK):
        n += len(ch)
        ch = ch.rename(columns={"PRES": "z", "TEMP": "T", "PSAL": "S", "LATITUDE": "lat",
                                "LONGITUDE": "lon", "TIME": "time"})
        for k in ("lon", "lat", "z", "T", "S"):
            ch[k] = pd.to_numeric(ch[k], errors="coerce")
        ch["pid"] = "ARGO|" + ch.PLATFORM_NUMBER.str.strip() + "|" + ch.CYCLE_NUMBER.str.strip()
        ch["instr"] = "Argo float"
        ch = ch[np.isfinite(ch.z) & (ch.z >= 0) & (ch.z <= 6000) & ch.PRES_QC.str.strip().isin(GOOD)]
        metas.append(ch.groupby("pid", sort=False).agg(
            time=("time", "first"), lon=("lon", "first"), lat=("lat", "first"),
            basin=("basin", "first"), instr=("instr", "first"), zmax=("z", "max")))
        sums["T"].append(_accumulate(ch[ch.TEMP_QC.str.strip().isin(GOOD) & np.isfinite(ch["T"])
                                        & (ch["T"] >= -5) & (ch["T"] <= 45)], "T"))
        sums["S"].append(_accumulate(ch[ch.PSAL_QC.str.strip().isin(GOOD) & np.isfinite(ch.S)
                                        & (ch.S >= 0) & (ch.S <= 60)], "S"))
        print(f"    Argo: {n/1e6:.0f}M rows, {time.time()-t0:.0f} s", flush=True)
    return _finish(sums, metas, "ARGO")


# ==========================================================
# MATCHING
# ==========================================================

def _agreement(s: pd.DataFrame, a: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Binned T and S of each SeaDataNet cast against its candidate twin."""
    st, at = s.set_index("pid"), a.set_index("pid")
    dT = np.abs(st.loc[pairs.pid, TCOLS].to_numpy(float) - at.loc[pairs.twin, TCOLS].to_numpy(float))
    dS = np.abs(st.loc[pairs.pid, SCOLS].to_numpy(float) - at.loc[pairs.twin, SCOLS].to_numpy(float))
    out = pairs.copy()
    okT = np.isfinite(dT)
    out["n_common_bins"] = okT.sum(1)
    out["shallow_common"] = okT[:, : BINS.index(100) + 1].any(1)
    with np.errstate(all="ignore"):
        out["max_abs_dT"] = np.where(okT.any(1), np.nanmax(np.where(okT, dT, np.nan), axis=1), np.nan)
        out["max_abs_dS"] = np.where(np.isfinite(dS).any(1),
                                     np.nanmax(np.where(np.isfinite(dS), dS, np.nan), axis=1), np.nan)
    return out


def find_copies(c: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    c["time"] = pd.to_datetime(c.time, format="mixed", errors="coerce")
    c = c.dropna(subset=["time"])
    a = c[c.source == "ARGO"].copy()
    ap = a.pid.str.split("|")
    a["wmo"], a["cyc"] = ap.str[1], ap.str[2].str.lstrip("0").replace("", "0")
    a["day"] = a.time.dt.floor("D")
    s = c[(c.source == "SDN") & (c.time >= a.time.min() - pd.Timedelta(days=2))].copy()
    sp = s.pid.str.split("|")
    s["cruise"], s["station"] = sp.str[2].fillna(""), sp.str[3].fillna("")
    s["float_named"] = s.cruise.str.contains(FLOAT_CRUISE, regex=True)

    # float, cycle and direction, where the entry names them
    pf = s.pid.str.extract(r"^SDN\|[^|]*\|PF_(\d+)\|\d+_(\d+)_([AD])$")
    ar = s.pid.str.extract(r"^SDN\|[^|]*\|Argo (\d+)\|(\d+)_[A-Z]_([AD])$")
    s["wmo"] = pf[0].fillna(ar[0])
    s["cyc"] = pf[1].fillna(ar[1]).str.lstrip("0").replace("", "0")
    s["direction"] = pf[2].fillna(ar[2])

    cols = ["pid", "time", "lon", "lat"]
    twin_cols = {"pid": "twin", "time": "ttime", "lon": "tlon", "lat": "tlat"}

    def _dist(j: pd.DataFrame) -> pd.DataFrame:
        j = j.copy()
        j["dt_h"] = (j.time - j.ttime).dt.total_seconds().abs() / 3600.0
        j["dpos_deg"] = np.maximum((j.lon - j.tlon).abs(), (j.lat - j.tlat).abs())
        return j

    found = []
    # --- id
    k = s[s.wmo.notna() & (s.direction == "A")]
    j = _dist(k[cols + ["wmo", "cyc"]].merge(a[cols + ["wmo", "cyc"]].rename(columns=twin_cols), on=["wmo", "cyc"]))
    j = j[j.dt_h <= ID_MAX_HOURS]
    found.append(j.assign(rule="id"))
    # --- candidates by day and position, for the two other rules
    cand = []
    for shift in (-1, 0, 1):
        aa = a[cols + ["day"]].rename(columns=twin_cols)
        aa["day"] = aa.day + pd.Timedelta(days=shift)
        deep = s[s.float_named | (s.zmax > 50)]
        j = deep[cols].assign(day=deep.time.dt.floor("D")).merge(aa, on="day")
        cand.append(j[((j.lon - j.tlon).abs() <= CONTENT_DEG) & ((j.lat - j.tlat).abs() <= CONTENT_DEG)])
    cand = _dist(pd.concat(cand, ignore_index=True)).drop_duplicates(["pid", "twin"])
    named = set(s.pid[s.float_named])
    descending = set(s.pid[s.direction == "D"])
    tp = cand[cand.pid.isin(named - descending) & (cand.dt_h <= TP_HOURS) & (cand.dpos_deg <= TP_DEG)]
    found.append(tp.assign(rule="time-position"))
    cc = _agreement(s, a, cand[~cand.pid.isin(descending)])
    cc = cc[(cc.n_common_bins >= CONTENT_MIN_BINS) & cc.shallow_common & (cc.max_abs_dT <= CONTENT_MAX_DT)]
    found.append(cc[cand.columns].assign(rule="content"))

    f = pd.concat(found, ignore_index=True)
    f["rank"] = f.rule.map({"id": 0, "time-position": 1, "content": 2})
    rules = f.groupby("pid").rule.agg(lambda x: "+".join(sorted(set(x), key=["id", "time-position", "content"].index)))
    best = f.sort_values(["pid", "rank", "dt_h", "dpos_deg"]).drop_duplicates("pid")
    best = _agreement(s, a, best[["pid", "twin", "dt_h", "dpos_deg"]])
    best["rules"] = best.pid.map(rules)

    out = s[s.float_named | s.pid.isin(set(best.pid))][
        ["pid", "basin", "time", "lon", "lat", "zmax", "instr", "cruise", "station", "float_named", "direction"]]
    out = out.merge(best.drop(columns="shallow_common"), on="pid", how="left")
    out["is_copy"] = out.twin.notna()
    out["year"] = out.time.dt.year
    return out.sort_values(["basin", "time", "pid"]).reset_index(drop=True)


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    for p in (SDN, ARGO):
        if not p.exists():
            sys.exit(f"missing input: {p}")
    t0 = time.time()
    print("building the per-cast tables from the master files ...", flush=True)
    c = pd.concat([build_sdn(), build_argo()], ignore_index=True)
    print(f"  {len(c):,} casts ({(c.source == 'SDN').sum():,} SeaDataNet, {(c.source == 'ARGO').sum():,} Argo), "
          f"{time.time() - t0:.0f} s", flush=True)

    out = find_copies(c)
    OUT_LIST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_LIST, index=False)
    copies = sorted(out.pid[out.is_copy])
    OUT_PIDS.write_text("\n".join(copies) + "\n")

    pd.set_option("display.width", 220)
    fam = out.cruise.str.replace(r"\d+", "#", regex=True).str.slice(0, 24)
    print(f"\n{len(out):,} SeaDataNet entries listed, {len(copies):,} with a twin in the Argo set\n")
    print("by cruise-name family:")
    print(out.assign(family=fam).groupby("family").agg(
        entries=("pid", "size"), copies=("is_copy", "sum"),
        descending=("direction", lambda x: int((x == "D").sum()))).sort_values("entries", ascending=False).to_string())
    print("\ncopies by matching rule(s):")
    print(out[out.is_copy].rules.value_counts().to_string())
    k = out[out.is_copy]
    print("\nagreement of the copies with their twins:")
    print(f"  time difference [h]:      median {k.dt_h.median():.3f}, 99 % {k.dt_h.quantile(.99):.2f}, max {k.dt_h.max():.2f}")
    print(f"  position difference [deg]: median {k.dpos_deg.median():.4f}, 99 % {k.dpos_deg.quantile(.99):.4f}, max {k.dpos_deg.max():.4f}")
    print(f"  largest binned |dT| [C]:  median {k.max_abs_dT.median():.4f}, 90 % {k.max_abs_dT.quantile(.9):.3f}, 99 % {k.max_abs_dT.quantile(.99):.3f}")
    print(f"  largest binned |dS|:      median {k.max_abs_dS.median():.4f}, 90 % {k.max_abs_dS.quantile(.9):.3f}, 99 % {k.max_abs_dS.quantile(.99):.3f}")
    print("\nfloat-named entries that are not copies:")
    n = out[out.float_named & ~out.is_copy]
    print(f"  {len(n):,}: descending profiles {int((n.direction == 'D').sum()):,}, "
          f"no such profile in the Argo set {int((n.direction != 'D').sum()):,}")
    print("\ncopies by basin:")
    print(k.groupby("basin").agg(copies=("pid", "size"), first=("year", "min"), last=("year", "max")).to_string())
    print(f"\nwrote {OUT_LIST}\nwrote {OUT_PIDS}\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
