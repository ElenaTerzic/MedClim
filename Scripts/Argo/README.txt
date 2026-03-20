Run the scripts in the following order:


1. safe_download_med_argo_regions.py

run like this - you can open 8 terminal windows in one go, so repeat this 3 times:

python safe_download_med_argo_regions.py --region-id 0
...
python safe_download_med_argo_regions.py --region-id 23


2. retry_failed_chunks_regions.py in case you have some undownloaded chunks in your log

3. merge_med_argo_chunks.py

4. qc_med_argo_phy_standard.py

5. nc_to_csv_all.py

6. argo_add_basins.py

7. argo_seasonal_climatology_TS.py --> save seasonal clmatologies for Manca's basins with the same method

8. plot_argo_like_profiles_per_basin.py --> plot seasonal climatology profiles


Then for post-proc you can first go for:

1. plot_basins_manca.py -> in order to check if you have points in each of the basins
2. map_argo_temporal_all.py -> to check the complete spatio-temporal distribution of your Argo data 
3.