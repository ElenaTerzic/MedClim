# Adriatic

This directory contains Adriatic-specific workflows used in the MedClim project.

## Directory structure

- `preprocessing/` - ingestion of Adriatic source datasets and export to long-format CSV
- `diagnostics/` - basin checks, outlier inspection, temporal coverage, and standard-depth diagnostics
- `SSL/` - salinity stratification layer analysis workflows
- `figures/` - plotting scripts for Adriatic-focused papers and reports

## Preprocessing

- `read_CTD_19_23.py`
- `read_IOR_Adriatic_database_1987_2019_big.py`
- `read_IOR_Palagruza_1952_2018.py`
- `read_IRB_Northern_Adriatic_Transect_1979_2018.py`

These scripts ingest source XLSX or netCDF datasets and export harmonized long-format CSV files for downstream use. The Palagruza, RB Northern Adriatic, and IOR readers assign Adriatic basin labels during processing.

## Diagnostics

- `check_standard_depths.py`
- `inspect_adriatic_out_basin.py`
- `inspect_adriatic_outliers_locations.py`
- `fix_adriatic_outliers_per_basin.py`
- `map_temporal_CTD_19_23.py`

These scripts validate basin assignment, inspect casts labeled outside DJ1-DJ3, check depth-grid alignment, and support spatial and temporal QC.

## SSL workflows

- `SSL_profiles.py`
- `SSL_profiles_dualmode_raw_or_binned.py`

These scripts perform per-cast SSL detection and seasonal statistics for selected Adriatic basins using SeaDataNet, Adriatic regional datasets, and Argo. The dual-mode script is used for raw-versus-binned SSL validation.

## Figures

- `plot_SSL_percent_per_year.py`
- `plot_SSLs_summary.py`
- `plot_DJ1_DJ3_publication_SIG0_maps.py`

These scripts generate Adriatic-specific publication figures and summary diagnostics.

## Notes

- Processed data are stored in the project data layout referenced by the scripts.
- The `SSL/` directory represents a distinct methodological sub-workflow within the broader MedClim project.
