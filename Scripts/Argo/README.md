# Argo

This directory contains the Mediterranean Argo-only workflow used in the MedClim project.

## Directory structure

- `acquisition/` – download, retry, and merge of raw Mediterranean Argo regional chunks
- `preprocessing/` – QC, NetCDF-to-CSV conversion, and Manca basin assignment
- `diagnostics/` – QC summaries, float-cycle checks, basin coverage, and temporal/depth diagnostics
- `climatology/` – seasonal climatology building and plotting
- `scratch_or_legacy/` – archived notes or non-essential legacy files

## Main workflow

1. `acquisition/safe_download_med_argo_regions.py`  
   Download Mediterranean Argo data in monthly regional chunks.

2. `acquisition/retry_failed_chunks_regions.py`  
   Retry failed downloads listed in the failure logs.

3. `acquisition/merge_med_argo_chunks.py`  
   Merge downloaded chunk NetCDF files into a single Mediterranean Argo dataset.

4. `preprocessing/qc_med_argo_phy_standard.py`  
   Apply physical-profile QC and compute derived variables.

5. `preprocessing/argo_nc_to_csv_all.py`  
   Convert the QCed merged NetCDF dataset into a profile-sorted CSV.

6. `preprocessing/argo_assign_manca_basins.py`  
   Assign Manca basins and export basin-specific CSV files.

7. `climatology/argo_seasonal_climatology_TS.py`  
   Build seasonal T/S climatological profiles per basin.

8. `climatology/plot_argo_like_profiles_per_basin.py`  
   Plot seasonal climatological profiles.

## Diagnostics

- `diagnostics/check_float_cycles_pipeline.py`
- `diagnostics/qc_diagnostics.py`
- `diagnostics/argo_depth_distribution.py`
- `diagnostics/map_argo_temporal_all.py`
- `diagnostics/plot_basins_manca.py`

## Notes

- Scripts currently use local hard-coded paths.
- Large raw/intermediate datasets are not tracked in Git.
- Outputs are typically written to project-level `Data/`, `Plots/`, `Results/`, or `Diagnostics/` directories.