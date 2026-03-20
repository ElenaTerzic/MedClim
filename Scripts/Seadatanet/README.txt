
Data source:

https://www.seadatanet.org/Products#/metadata/2a2aa0c5-4054-4a62-a18b-3835b304fe64

Download link:

https://files.seadatanet.org/aggregated_datasets/Med_Sea/SDC_MED_DATA_TS_V2/

Export the data to .csv with Ocean Data Viewer (ODV)

Use these codes:


1. clean_seadatanet_odv_export.py         -->  clean the data from metadata rubbish, uniform varnames and get a clean .csv

2. stats_depth_coverage_profiles_only.py  -->  replicate the binned histogram to check with the report:

                                                  https://archimer.ifremer.fr/doc/00658/77059/78361.pdf

3. make_level_maps_TS.py                  -->  do a spatial map for certain depths to check consistency with the report

4. barplot_profiles_per_year_platform.py  -->  check that the number of profiles per platform annually is  consistent with     
                                               the report and also similar to Manca's figure 3

5. add_basins.py                          -->  do a basin division and save the .csv

6. manca_seasonal_climatology_TS.py       --> obtain seasonal climatologies and store them in the .csv

7. plot_manca_like_profiles_per_basin.py  --> do the plots of seasonal profiles like in the original paper


