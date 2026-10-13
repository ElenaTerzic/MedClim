# Seadatanet

This directory contains the SeaDataNet-only workflow used in the MedClim project.

## Directory structure

- `preprocessing/` – ODV export cleaning and Manca basin assignment
- `diagnostics/` – platform, depth, and temporal diagnostics
- `climatology/` – seasonal climatology building
- `figures/` – profile and level-map plotting
- `outputs/` – generated diagnostic tables (not source code)
- `scratch_or_legacy/` – archived notes or obsolete files

## Main workflow

1. `preprocessing/clean_seadatanet_odv_export.py`  
   Clean raw ODV CSV export, normalize headers, and create cleaned/profile-only files.

2. `diagnostics/stats_depth_coverage_profiles_only.py`  
   Evaluate depth coverage by cast.

3. `figures/make_level_maps_TS.py`  
   Build T/S level maps for selected depths.

4. `diagnostics/barplot_profiles_per_year_platform.py`  
   Summarize casts per year by platform.

5. `preprocessing/seadatanet_assign_manca_basins.py`  
   Assign Manca basins and export basin-tagged CSV files.

6. `preprocessing/identify_argo_copies_in_seadatanet.py`  
   After the coastal mask (`MultiPlatform/filtering_qc/remove_coastline_data.py`) and the
   Argo preprocessing: list the SeaDataNet casts that are Argo float profiles already held
   by the Argo set (SeaDataNet archives float cycles as casts under cruises `PF_<WMO>`,
   `Argo <WMO>`, `ARGO-ITALY …`, a few as "CTD" under `SHODB`). Matching by float id and
   cycle, by time and position, and by identical binned temperatures. Writes
   `Data/Seadatanet/sdn_argo_copies.csv` and the tracked `preprocessing/sdn_argo_copies_pids.txt`
   (21,926 casts, 2005–2019).

7. `preprocessing/remove_argo_copies_from_seadatanet.py`  
   Write the SeaDataNet master file (and its per-cast diagnostics file) without the listed
   casts: `*_no_argo_copies.csv`. All downstream MultiPlatform / Publication_Papers scripts
   read these files.

8. `climatology/manca_seasonal_climatology_TS.py`  
   Build seasonal climatologies by basin.

9. `figures/plot_manca_like_profiles_per_basin.py`  
   Plot seasonal climatological profiles.

## Additional diagnostics

- `diagnostics/analyse_data_depths.py`
- `diagnostics/map_temporal_by_platform_centuries.py`
- `diagnostics/inspect_sdn_columns_platforms.py`
- `diagnostics/inspect_sdn_columns_platforms_v2.py`
- `diagnostics/inspect_sdn_platform_castcounts.py`
- `diagnostics/sdn_diagnostic_like_table3.py`
- `diagnostics/sdn_debug_platform_spikes.py`

## Notes

- Paths are resolved through `Scripts/paths.py` (see the repository README).
- Large generated CSV diagnostics should not be tracked in Git.
- Data files are stored outside the repository.