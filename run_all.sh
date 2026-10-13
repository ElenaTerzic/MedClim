#!/usr/bin/env bash
# MedClim pipeline, in the order of README.md ("Main workflow" 1-9, then the manuscript
# figures).  Every script resolves its paths through Scripts/paths.py: data and outputs
# live under the repository (Data/, Stats/, Plots/, Results/, Diagnostics/) or under
# $MEDCLIM_DATA when that is set.  The Basins package must be importable and its checkout
# findable (see Scripts/paths.py); JADMON 2024 needs $MEDCLIM_JADMON_2024.
#
#   ./run_all.sh              run every stage in order
#   ./run_all.sh trends figs  run only the named stages
#   ./run_all.sh --list       show the stage names
#
# Stages 1-3 need the source data (a SeaDataNet ODV export, the Argo GDAC over the network,
# the Adriatic source files), which this repository does not redistribute.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

run() { echo; echo "== $*"; "$PY" "$@"; }

stage_seadatanet() {                       # README step 1
  run Scripts/Seadatanet/preprocessing/clean_seadatanet_odv_export.py
  run Scripts/Seadatanet/preprocessing/seadatanet_assign_manca_basins.py
}
stage_argo() {                             # README step 2
  run Scripts/Argo/acquisition/safe_download_med_argo_regions.py   # all regions; or --region-id N per tile
  run Scripts/Argo/acquisition/retry_failed_chunks_regions.py
  run Scripts/Argo/acquisition/merge_med_argo_chunks.py
  run Scripts/Argo/preprocessing/qc_med_argo_phy_standard.py
  run Scripts/Argo/preprocessing/argo_nc_to_csv_all.py
  run Scripts/Argo/preprocessing/argo_assign_manca_basins.py
}
stage_adriatic() {                         # README step 3: the four readers behind the Zenodo data record
  run Scripts/Adriatic/preprocessing/read_CTD_19_23.py
  run Scripts/Adriatic/preprocessing/read_IOR_Adriatic_database_1987_2019_big.py
  run Scripts/Adriatic/preprocessing/read_IOR_Palagruza_1952_2018.py
  run Scripts/Adriatic/preprocessing/read_IRB_Northern_Adriatic_Transect_1979_2018.py
  # run Scripts/Adriatic/preprocessing/read_JADMON_2024.py          # 2024 delivery, not part of the GRL analysis
  # The readers write their *_long.csv tables next to their figures under Plots/Adriatic/;
  # place them in Data/Adriatic/Processed/ before this step, which writes the *_long_fixed.csv
  # tables the later stages read.
  run Scripts/Adriatic/diagnostics/fix_adriatic_outliers_per_basin.py
}
stage_coast() {                            # README step 4
  run Scripts/MultiPlatform/filtering_qc/remove_coastline_data.py
}
stage_argo_copies() {                      # README step 5
  run Scripts/Seadatanet/preprocessing/identify_argo_copies_in_seadatanet.py
  run Scripts/Seadatanet/preprocessing/remove_argo_copies_from_seadatanet.py
}
stage_climatology() {                      # README step 6 (RUN_BASIN_MODES / RUN_PERIODS select subsets)
  run Scripts/MultiPlatform/core/multi_source_seasonal_climatology.py
}
stage_monthly() {                          # README step 7
  run Scripts/MultiPlatform/core/build_monthly_timeseries_all_bins.py
  run Scripts/MultiPlatform/core/build_monthly_timeseries_composed_from_atomic.py
}
stage_trends() {                           # README step 8
  run Scripts/MultiPlatform/core/fit_trends_all_bins.py --mode atomic
  run Scripts/MultiPlatform/core/fit_trends_all_bins.py --mode composed
}
stage_metadata() {                         # README step 9
  run Scripts/MultiPlatform/core/save_metadata.py
}
stage_figs() {                             # README "Manuscript figures"
  run Scripts/Publication_Papers/dataset_coverage.py                          # Figure 1
  run Scripts/Publication_Papers/century_differences_composite.py             # Figure 2
  run Scripts/Publication_Papers/maps_trends.py                               # Figure 3
  run Scripts/Publication_Papers/vertical_trends_heatmaps.py                  # Figure 4
  run Scripts/Publication_Papers/supplement_vertical_trend_bars_other_basins.py
  run Scripts/Additional/robustness_model_comparison.py
  run Scripts/Additional/sensitivity_test_trends.py --mode composed
}

STAGES=(seadatanet argo adriatic coast argo_copies climatology monthly trends metadata figs)

if [[ "${1:-}" == "--list" ]]; then printf '%s\n' "${STAGES[@]}"; exit 0; fi
if [[ $# -gt 0 ]]; then STAGES=("$@"); fi
for s in "${STAGES[@]}"; do
  declare -F "stage_$s" >/dev/null || { echo "unknown stage: $s (see --list)" >&2; exit 2; }
  "stage_$s"
done
echo; echo "done: ${STAGES[*]}"
