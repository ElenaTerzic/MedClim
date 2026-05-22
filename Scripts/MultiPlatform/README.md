# MultiPlatform

This directory contains the core multi-source Mediterranean climatology and trend-analysis workflow used in the MedClim project.

## Purpose

The `MultiPlatform` workflow combines several observational data sources into a common basin-based analysis framework using Manca-style Mediterranean sub-basins. It supports:

- coastal and estuarine filtering
- seasonal climatology building using standard-depth-centered bins
- monthly time-series construction by basin, depth, and variable
- trend fitting for 20th-century, 21st-century, and full-period analyses
- support diagnostics and publication figures

## Structure

- `core/` - main climatology, monthly-series, metadata, and trend scripts
- `filtering_qc/` - coastal filtering and exclusion-mask diagnostics
- `sampling_qc/` - vertical bin support and cast-count diagnostics
- `figures/` - plotting scripts for maps, profiles, Hovmoller diagrams, and T-S diagrams

## Main workflow

1. Coastal and estuarine filtering
   `remove_coastline_data.py` removes casts influenced by mainland proximity, inland positions, and manually tuned river or strait masks.

2. Filtering diagnostics
   `plot_exclusion_masks_and_points.py` visualizes retained and excluded observations.

3. Seasonal climatology building
   `multi_source_seasonal_climatology.py` builds seasonal climatologies using standard-depth-centered bins and per-cast bin means.

4. Monthly time-series building
   `build_monthly_timeseries_all_bins.py` creates basin x depth x variable monthly series.
   `build_monthly_timeseries_composed_from_atomic.py` aggregates atomic-basin series into composed regions.

5. Trend fitting
   `fit_trends_all_bins.py` fits deseasoned linear trends for the 20th and 21st centuries and quadratic full-period trends and accelerations.

6. Metadata and support products
   `save_metadata.py` rebuilds the point metadata tables used by the map panels and support figures.

7. Sampling diagnostics
   `bin_coverage_centered_on_STD_DEPTHS.py`, `analyse_bin_coverage.py`, `plot_cast_counts_profile_by_basin.py`, and `plot_vertical_res.py` evaluate vertical data support and bin eligibility.

8. Figure generation
   Representative scripts include `plot_manca_all_basins.py`, `plot_manca_delta_overall_maps.py`, `plot_manca_delta_seasonal_maps.py`, `plot_linear_trends_20c_vs_21c_maps.py`, `plot_quad_trend_accel_maps.py`, `plot_vertical_trend_bars_composed.py`, `plot_hovmoeller.py`, `plot_T_S_diagrams.py`, and `plot_map_with_basins.py`.

## Notes

- Outputs are generated into project-level `Plots/`, `Results/`, `Stats/`, and `Diagnostics/` directories.
- The main manuscript figure scripts live in `Scripts/Publication_Papers/` and build on the products created here.
