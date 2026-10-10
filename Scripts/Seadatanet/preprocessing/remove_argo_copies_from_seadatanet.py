#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write the SeaDataNet master file without the copies of Argo profiles.

Reads the cast ids listed by identify_argo_copies_in_seadatanet.py (the tracked
sdn_argo_copies_pids.txt, or the full Data/Seadatanet/sdn_argo_copies.csv when present)
and streams the SeaDataNet master file, dropping every row of a listed cast:

  Data/Seadatanet/Seadatanet.profiles_only_with_basins_mainland_special_excluded.csv
    -> Data/Seadatanet/Seadatanet.profiles_only_with_basins_mainland_special_excluded_no_argo_copies.csv

The per-cast diagnostics file of the coastal mask (one row per cast, used for the
Figure 1 map) is filtered the same way:

  ..._profile_diagnostics.csv -> ..._no_argo_copies_profile_diagnostics.csv

All downstream scripts (climatology, monthly series, cast counts, metadata, Figure 1)
read the *_no_argo_copies files.  The cast id is the pipeline's
"SDN|<LOCAL_CDI_ID>|<Cruise>|<Station>"; the diagnostics file keys casts as
"SDN||<LOCAL_CDI_ID>||<Cruise>||<Station>".
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "Data/Seadatanet"

MASTER = DATA / "Seadatanet.profiles_only_with_basins_mainland_special_excluded.csv"
MASTER_OUT = DATA / "Seadatanet.profiles_only_with_basins_mainland_special_excluded_no_argo_copies.csv"
DIAG = DATA / "Seadatanet.profiles_only_with_basins_mainland_special_excluded_profile_diagnostics.csv"
DIAG_OUT = DATA / "Seadatanet.profiles_only_with_basins_mainland_special_excluded_no_argo_copies_profile_diagnostics.csv"

LIST_CSV = DATA / "sdn_argo_copies.csv"
LIST_TXT = Path(__file__).resolve().parent / "sdn_argo_copies_pids.txt"

CDI, CRUISE, STATION = "LOCAL_CDI_ID", "Cruise", "Station:METAVAR:INDEXED_TEXT"
CHUNK = 1_000_000


def load_copies() -> set:
    if LIST_CSV.exists():
        d = pd.read_csv(LIST_CSV, usecols=["pid", "is_copy"])
        pids = set(d.pid[d.is_copy])
        print(f"copy list: {LIST_CSV} ({len(pids):,} casts)")
    elif LIST_TXT.exists():
        pids = {l.strip() for l in LIST_TXT.read_text().splitlines() if l.strip()}
        print(f"copy list: {LIST_TXT} ({len(pids):,} casts)")
    else:
        sys.exit("no copy list found; run identify_argo_copies_in_seadatanet.py first")
    return pids


def filter_master(copies: set) -> set:
    t0 = time.time()
    n_in = n_out = 0
    seen = set()
    header = True
    MASTER_OUT.unlink(missing_ok=True)
    for ch in pd.read_csv(MASTER, dtype=str, chunksize=CHUNK, low_memory=False):
        n_in += len(ch)
        pid = "SDN|" + ch[CDI].fillna("") + "|" + ch[CRUISE].fillna("") + "|" + ch[STATION].fillna("")
        drop = pid.isin(copies)
        seen.update(pid[drop].unique())
        keep = ch[~drop]
        keep.to_csv(MASTER_OUT, mode="a", index=False, header=header)
        header = False
        n_out += len(keep)
        print(f"    {n_in/1e6:.0f}M rows read, {n_in - n_out:,} dropped, {time.time()-t0:.0f} s", flush=True)
    print(f"master: {n_in:,} rows -> {n_out:,} rows; {n_in - n_out:,} rows of {len(seen):,} casts removed")
    return seen


def filter_diagnostics(copies: set) -> None:
    if not DIAG.exists():
        print(f"(no diagnostics file at {DIAG}; skipped)")
        return
    uids = {"SDN||" + p[len("SDN|"):].replace("|", "||") for p in copies}
    d = pd.read_csv(DIAG, dtype=str, low_memory=False)
    drop = d["_cast_uid"].isin(uids)
    d[~drop].to_csv(DIAG_OUT, index=False)
    print(f"diagnostics: {len(d):,} casts -> {int((~drop).sum()):,}; {int(drop.sum()):,} removed")


def main() -> None:
    if not MASTER.exists():
        sys.exit(f"missing input: {MASTER}")
    copies = load_copies()
    seen = filter_master(copies)
    missing = copies - seen
    print(f"listed copies: {len(copies):,}; met in the master file: {len(seen):,}; not met: {len(missing):,}")
    if missing:
        print("  first not met:", sorted(missing)[:5])
    filter_diagnostics(copies)
    print(f"wrote {MASTER_OUT}\nwrote {DIAG_OUT}")


if __name__ == "__main__":
    main()
