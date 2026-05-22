# Seadatanet

This directory contains the SeaDataNet-only workflow used in the MedClim project.

## Directory structure

- `preprocessing/` - ODV export cleaning and Manca basin assignment
- `diagnostics/` - platform, depth, and temporal diagnostics
- `climatology/` - seasonal climatology building
- `figures/` - profile and level-map plotting
- `outputs/` - generated diagnostic tables that are not part of the tracked source code

## Main workflow

1. `preprocessing/clean_seadatanet_odv_export.py`
   Clean the raw ODV CSV export, normalize headers, and create cleaned and profile-only files.

2. `diagnostics/stats_depth_coverage_profiles_only.py`
   Evaluate depth coverage by cast.

3. `preprocessing/seadatanet_assign_manca_basins.py`
   Assign Manca basins and export basin-tagged CSV files.

4. `climatology/manca_seasonal_climatology_TS.py`
   Build seasonal climatologies by basin.

5. `figures/plot_manca_like_profiles_per_basin.py`
   Plot seasonal climatological profiles.

6. `figures/make_level_maps_TS.py`
   Build temperature and salinity level maps for selected depths.

## Additional diagnostics

- `diagnostics/analyse_data_depths.py`
- `diagnostics/map_temporal_by_platform_centuries.py`
- `diagnostics/inspect_sdn_columns_platforms.py`
- `diagnostics/inspect_sdn_columns_platforms_v2.py`
- `diagnostics/inspect_sdn_platform_castcounts.py`
- `diagnostics/sdn_diagnostic_like_table3.py`
- `diagnostics/sdn_debug_platform_spikes.py`

## Notes

- Large generated CSV diagnostics are not tracked in Git.
- SeaDataNet input files are expected in the project data layout used by the processing scripts.
