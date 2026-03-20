# Adriatic

This directory contains Adriatic-specific workflows used in the MedClim project.

## Directory structure

- `preprocessing/` – ingestion of Adriatic source datasets and export to long-format CSV
- `diagnostics/` – basin checks, outlier inspection, temporal coverage, and standard-depth diagnostics
- `SSL/` – salinity stratification layer (SSL) analysis workflows
- `figures/` – plotting scripts for Adriatic-focused papers and reports

## Preprocessing

- `read_CTD_19_23.py`
- `read_IOR_Adriatic_database_1987_2019_big.py`
- `read_IOR_Palagruza_1952_2018.py`
- `read_IRB_Northern_Adriatic_Transect_1979_2018.py`

These scripts ingest source XLSX/netCDF datasets and export harmonized long-format CSV files for downstream use. The Palagruža, RB Northern Adriatic, and IOR readers assign Adriatic basin labels during processing. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

## Diagnostics

- `check_standard_depths.py`
- `inspect_adriatic_out_basin.py`
- `inspect_adriatic_outliers_locations.py`
- `fix_adriatic_outliers_per_basin.py`
- `map_temporal_CTD_19_23.py`

These scripts validate basin assignment, inspect casts labeled outside DJ1–DJ3, check depth-grid alignment, and support spatial/temporal QC. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

## SSL workflows

- `SSL_profiles.py`
- `SSL_profiles_dualmode_raw_or_binned.py`

These scripts perform per-cast SSL detection and seasonal statistics for selected Adriatic basins, using a multi-source input framework (SeaDataNet, Adriatic regional datasets, and Argo). The dual-mode script is used for validation of raw-vs-binned SSL detection. :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}

## Figures

- `plot_SSL_percent_per_year.py`
- `plot_SSLs_summary.py`
- `plot_DJ1_DJ3_publication_SIG0_maps.py`

These scripts generate Adriatic-specific publication figures and summary diagnostics. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

## Notes

- Scripts currently use local hard-coded paths.
- Processed data are stored outside the repository.
- The `SSL/` directory corresponds to a distinct methodological sub-workflow and may later be split into a separate paper-specific workflow.