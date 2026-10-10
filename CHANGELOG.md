# Changelog

## 2026-10-10 — remove SeaDataNet copies of Argo profiles

**What.** The SeaDataNet aggregation SDC_MED_DATA_TS_V2 archives Argo float profiles as
casts of its own (cruises `PF_<WMO>`, `Argo <WMO>`, `ARGO-ITALY …`, and a few as "CTD"
under cruise `SHODB`). The pipeline removed duplicates within a source only, so 21,926
SeaDataNet casts (2005–2019, every Manca basin except DJ1) that are also in the Argo set
entered every pooled 2000–2025 product twice. The problem was found on 5 October 2026
during the Ocean Science (Adriatic) revision.

**Fix.** Two new preprocessing scripts, run after the coastal mask:

- `Scripts/Seadatanet/preprocessing/identify_argo_copies_in_seadatanet.py` builds per-cast
  tables from the SeaDataNet and Argo master files and lists the SeaDataNet casts that have
  an Argo twin (same float and cycle within 24 h; or float-named and within 3 h / 0.05°;
  or identical binned temperatures within one day / 0.1°). Descending float profiles and
  float profiles the Argo set does not hold are not copies and stay. Output:
  `Data/Seadatanet/sdn_argo_copies.csv` and the tracked list
  `Scripts/Seadatanet/preprocessing/sdn_argo_copies_pids.txt`.
- `Scripts/Seadatanet/preprocessing/remove_argo_copies_from_seadatanet.py` writes the
  SeaDataNet master file and its per-cast diagnostics file without those casts
  (`*_no_argo_copies.csv`).

All analysis scripts now read the `*_no_argo_copies` files: the seasonal climatology,
the monthly series (and therefore all trends), the cast-count and sampling-support
diagnostics, the metadata tables and Figure 1. The per-year SeaDataNet cast counts of
Figure 1b are now taken from the deduplicated analysis master file (basin-assigned,
coast-masked) instead of the raw `Seadatanet.profiles_only.csv`, so that figure counts
the casts that enter the analysis. The coastal-mask QC tools under
`Scripts/MultiPlatform/filtering_qc/` still read the pre-deduplication diagnostics file,
since they tune a step that precedes the deduplication.

**Effect on the published results** (Terzić & Vilibić, GRL 2026; audited 6–9 October 2026
by re-running the unchanged pipeline without the copies). Full-period trends and
accelerations change by a median of 0.0004 °C/decade and 0.0002 °C/decade² (about 5 % of
one standard error); no ranking, sign or significance statement changes. Century
differences move by up to 0.08 °C at single map cells (Ionian 200 m warming 0.92 → 0.96 °C,
northern Aegean surface salinity difference 1.10 → 1.05). The seasonal-climatology values
quoted in the text move at the second decimal (NWM summer 20 m warming 2.01 → 1.95 °C,
Ionian autumn 5 m 3.19 → 3.23 °C, Ionian winter 40 m salinity 0.58 → 0.60, NWM summer
20 m density −0.52 → −0.51) with peak seasons, depths and rankings unchanged. The Ionian
T–S median shifts become 1.30 °C and 0.27 (published 1.27 and 0.26). The six Figure 1
anomaly peaks are unchanged; the Figure 1b SeaDataNet bars for 2014–2019 were too high.
The products archived with the paper were produced with the copies included.
