# MultiPlatform

This directory contains the core multi-source Mediterranean climatology and trend-analysis workflow used in the MedClim project.

## Purpose

The `MultiPlatform` workflow combines several observational data sources (e.g. Argo, SeaDataNet, Adriatic datasets) into a common basin-based analysis framework using Manca-style Mediterranean sub-basins. It supports:

- coastal/estuarine filtering and spatial QC
- seasonal climatology building using standard-depth-centered bins
- monthly time-series construction by basin, depth, and variable
- trend fitting (linear 20th vs 21st century, and quadratic full-period)
- support diagnostics and publication figures

## Suggested structure

- `core/` – main climatology, monthly-series, metadata, and trend scripts
- `filtering_qc/` – coastal filtering and exclusion-mask diagnostics
- `sampling_qc/` – vertical bin support and cast-count diagnostics
- `figures/` – plotting scripts for maps, profiles, Hovmöller diagrams, and T–S diagrams

## Main workflow

### 1. Coastal / estuarine filtering
- `remove_coastline_data.py`  
  Removes casts influenced by mainland proximity, inland positions, and manually tuned river/strait masks.

- `plot_exclusion_masks_and_points.py`  
  Visual QC of filtering masks and excluded/retained points.

- `test_estuary_locations.py`  
  Tuning utility for candidate river-mouth and strait masks.

### 2. Climatology building
- `multi_source_seasonal_climatology.py`  
  Builds seasonal climatologies using standard-depth-centered bins and per-cast bin means.
  Supports both atomic and composed basin modes.

### 3. Monthly time-series building
- `build_monthly_timeseries_all_bins.py`  
  Builds monthly basin × depth × variable time series from the filtered source datasets.

- `build_monthly_timeseries_composed_from_atomic.py`  
  Aggregates atomic-basin monthly series into composed Mediterranean regions.

### 4. Trend fitting
- `fit_trends_all_bins.py`  
  Fits deseasoned linear trends for the 20th and 21st centuries and quadratic full-period trends/accelerations.

### 5. Metadata and support products
- `save_metadata.py`  
  Rebuilds point metadata tables used for basin maps and support panels.

### 6. Sampling diagnostics
- `bin_coverage_centered_on_STD_DEPTHS.py`
- `analyse_bin_coverage.py`
- `plot_cast_counts_profile_by_basin.py`
- `plot_vertical_res.py`

These scripts evaluate vertical data support, bin eligibility, and cast-count structure.

### 7. Figure generation
Examples:
- `plot_manca_all_basins.py`
- `plot_manca_delta_overall_maps.py`
- `plot_manca_delta_seasonal_maps.py`
- `plot_linear_trends_20c_vs_21c_maps.py`
- `plot_quad_trend_accel_maps.py`
- `plot_vertical_trend_bars_composed.py`
- `plot_hovmoeller.py`
- `plot_T_S_diagrams.py`
- `plot_map_with_basins.py`

## Notes

- Paths are resolved through `Scripts/paths.py` (see the repository README).
- Several outputs are generated into project-level `Plots/`, `Results/`, `Stats/`, and `Diagnostics/` directories.
- Before publishing, deprecated scripts, temporary outputs, and `__pycache__` should be removed or ignored.