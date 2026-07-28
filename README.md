# MedClim

Mediterranean hydrographic processing and visualization workflows used for the analysis of
accelerating Mediterranean warming and salinification (Terzić & Vilibić, *Geophysical
Research Letters*).

## Repository scope

This repository is organized around the tracked analysis scripts used to preprocess source
datasets, build basin-based climatologies and time series, fit trends, and generate
manuscript figures. Large source data files and most generated products are intentionally
excluded from version control.

## Main workflow

1. `Scripts/Seadatanet/preprocessing/clean_seadatanet_odv_export.py` cleans the SeaDataNet
   export and prepares basin-tagged profile files.
2. `Scripts/Argo/acquisition/` and `Scripts/Argo/preprocessing/` download, quality-control,
   and basin-tag Mediterranean Argo profiles.
3. `Scripts/Adriatic/preprocessing/` ingests the Adriatic regional datasets and exports
   harmonized long-format tables. These four readers are the provenance for the cleaned
   Adriatic datasets deposited in the accompanying Zenodo data record.
4. `Scripts/MultiPlatform/filtering_qc/remove_coastline_data.py` applies the coastal and
   estuarine exclusion masks used before pooled analysis.
5. `Scripts/MultiPlatform/core/multi_source_seasonal_climatology.py` builds the seasonal
   climatology products.
6. `Scripts/MultiPlatform/core/build_monthly_timeseries_all_bins.py` and
   `Scripts/MultiPlatform/core/build_monthly_timeseries_composed_from_atomic.py` create
   monthly basin-by-depth series.
7. `Scripts/MultiPlatform/core/fit_trends_all_bins.py` fits the linear and quadratic trend
   products used in the manuscript.
8. `Scripts/MultiPlatform/core/save_metadata.py` rebuilds map-support metadata tables.

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
- Several scripts expect the local data layout used during manuscript production. The
  workflow documentation in `Scripts/*/README.md` identifies which scripts were used for
  each stage.
- The mapping scripts read basin polygons from a sibling `Basins` checkout, resolved
  relative to the repository root.

## Data availability

Argo and SeaDataNet source data are publicly available from their original repositories and
are cited in the manuscript; they are not redistributed here. The cleaned regional Adriatic
in situ datasets are deposited in a separate Zenodo record, produced by the readers in
`Scripts/Adriatic/preprocessing/`.

## License

Distributed under the Creative Commons Attribution 4.0 International license (CC BY 4.0).
