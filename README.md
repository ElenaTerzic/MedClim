# MedClim

Mediterranean hydrographic processing and visualization workflows used for the analysis of
accelerating Mediterranean warming and salinification (Terzić & Vilibić, *Geophysical
Research Letters*).

## Repository scope

This repository is organized around the tracked analysis scripts used to preprocess source
datasets, build basin-based climatologies and time series, fit trends, and generate
manuscript figures. Large source data files and most generated products are intentionally
excluded from version control.

## Running the code

- `environment.yml` lists the Python packages, pinned to the versions the manuscript
  products were made with (`conda env create -f environment.yml`). It also installs the
  two packages that are not on PyPI, `Basins` (basin polygons) and `MapPlotter`
  (satellite basemaps), from GitHub.
- Every script resolves its input and output locations through `Scripts/paths.py`. The
  repository root is taken from that file's own location, and the data trees `Data/`,
  `Stats/`, `Plots/`, `Results/` and `Diagnostics/` live under the repository unless the
  environment variable `MEDCLIM_DATA` points somewhere else. The `Basins` package's
  `manca.py` and the JADMON 2024 delivery are the only inputs outside the repository;
  `Scripts/paths.py` says where they are expected and which variables (`MEDCLIM_BASINS`,
  `MEDCLIM_JADMON_2024`) override that.
- `run_all.sh` runs the stages of the main workflow below in order, or a named selection
  of them (`./run_all.sh --list`).
- Each script starts with two comment lines, `Reads` and `Writes`, naming its inputs and
  outputs relative to the repository.
- `CITATION.cff` gives the citation for the code (Zenodo concept DOI
  10.5281/zenodo.20410480) and for the article.

## Main workflow

1. `Scripts/Seadatanet/preprocessing/clean_seadatanet_odv_export.py` cleans the SeaDataNet
   export and prepares basin-tagged profile files.
2. `Scripts/Argo/acquisition/` and `Scripts/Argo/preprocessing/` download, quality-control,
   and basin-tag Mediterranean Argo profiles.
3. `Scripts/Adriatic/preprocessing/` ingests the Adriatic regional datasets and exports
   harmonized long-format tables. These four readers are the provenance for the cleaned
   Adriatic datasets deposited in the accompanying Zenodo data record.
   `Scripts/Adriatic/diagnostics/fix_adriatic_outliers_per_basin.py` then reassigns the
   casts the readers left outside the three Adriatic basins (`BASIN_3ADR == 'out'`) to
   DJ1/DJ2/DJ3 by nearest centroid and writes the `*_long_fixed.csv` tables in
   `Data/Adriatic/Processed/` that the later stages read.
4. `Scripts/MultiPlatform/filtering_qc/remove_coastline_data.py` applies the coastal and
   estuarine exclusion masks used before pooled analysis.
5. `Scripts/Seadatanet/preprocessing/identify_argo_copies_in_seadatanet.py` lists the
   SeaDataNet casts that are Argo float profiles already held by the Argo set (the
   SeaDataNet aggregation archives float cycles as casts of its own), and
   `Scripts/Seadatanet/preprocessing/remove_argo_copies_from_seadatanet.py` writes the
   SeaDataNet master file without them (`*_no_argo_copies.csv`), which every downstream
   script reads. The list of the 21,926 affected casts is tracked as
   `Scripts/Seadatanet/preprocessing/sdn_argo_copies_pids.txt`. See `CHANGELOG.md`.
6. `Scripts/MultiPlatform/core/multi_source_seasonal_climatology.py` builds the seasonal
   climatology products.
7. `Scripts/MultiPlatform/core/build_monthly_timeseries_all_bins.py` and
   `Scripts/MultiPlatform/core/build_monthly_timeseries_composed_from_atomic.py` create
   monthly basin-by-depth series.
8. `Scripts/MultiPlatform/core/fit_trends_all_bins.py` fits the linear and quadratic trend
   products used in the manuscript.
9. `Scripts/MultiPlatform/core/save_metadata.py` rebuilds map-support metadata tables.

## Manuscript figures

- Figure 1: `Scripts/Publication_Papers/dataset_coverage.py`
- Figure 2: `Scripts/Publication_Papers/century_differences_composite.py`
- Figure 3: `Scripts/Publication_Papers/maps_trends.py`
- Figure 4: `Scripts/Publication_Papers/vertical_trends_heatmaps.py`
- Supplementary vertical-structure figures:
  `Scripts/Publication_Papers/supplement_vertical_trend_bars_other_basins.py`
- Robustness checks: `Scripts/Additional/robustness_model_comparison.py` and
  `Scripts/Additional/sensitivity_test_trends.py`

## Notes

- The tracked repository is script-focused. Large raw inputs and most derived data files
  are ignored to keep the repository lightweight.
- The data layout is the one used during manuscript production, described in
  `Scripts/paths.py`. The workflow documentation in `Scripts/*/README.md` identifies which
  scripts were used for each stage.
- The mapping scripts parse basin rectangles from the `Basins` package's own `manca.py`,
  located by `Scripts/paths.py` (installed package first, or a checkout named by
  `MEDCLIM_BASINS`).

## Data availability

Argo and SeaDataNet source data are publicly available from their original repositories and
are cited in the manuscript; they are not redistributed here. The cleaned regional Adriatic
in situ datasets are deposited in a separate Zenodo record, produced by the readers in
`Scripts/Adriatic/preprocessing/`.

## License

Distributed under the Creative Commons Attribution 4.0 International license (CC BY 4.0).
