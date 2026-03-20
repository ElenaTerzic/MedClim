#!/usr/bin/env python

# Divide the data set in regions - use the original .csv, saved with nc_to_csv_all.py

import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import numpy as np

import Basins

def filter_rows_by_values(df, col, values):
	return df[~df[col].isin(values)]


INPUTDIR     = "/Users/elena/Projects/IRB/MedClim/Data/Argo/Merged/"
OUTDIR       = "/Users/elena/Projects/IRB/MedClim/Data/Argo/Basins/"

df        = pd.read_csv(os.path.join(INPUTDIR, 'argo_med_all_sorted.csv'))


# Debug basins
basin_list = [Basins.manca.df2, Basins.manca.df3, Basins.manca.df4, Basins.manca.ds2,Basins.manca.df1,
			  Basins.manca.ds1, Basins.manca.ds3, Basins.manca.ds4, Basins.manca.dt1,Basins.manca.dt2,
			  Basins.manca.dt3, Basins.manca.di1, Basins.manca.di3, Basins.manca.dj1, Basins.manca.dj2, 
			  Basins.manca.dj3, Basins.manca.dj6, Basins.manca.dj4, Basins.manca.dj7, Basins.manca.dj8, 
			  Basins.manca.dj5, Basins.manca.dh1, Basins.manca.dh2, Basins.manca.dh3, Basins.manca.dl1, 
			  Basins.manca.dl2, Basins.manca.dl3, Basins.manca.dl4]

df['basin']  = 'none'

for ibasin, basin in enumerate(basin_list):

	print(ibasin, basin.name)

	xyz         = np.zeros((df.shape[0], 3))

	xyz[:,0]    = df.LONGITUDE
	xyz[:,1]    = df.LATITUDE

	mask        = basin.areinside(xyz)

	df['basin'].iloc[mask] = basin.abbrev 

# Exclude the None
df_out = df[df['basin'] != 'none']

print('There are now %d floats in total.'%(df_out.PLATFORM_NUMBER.unique().shape[0]))

# Break the data frame also to separate sub-basins to speed up the runs

df_basin = df_out.groupby(df_out['basin'])

for basin_name in df_out['basin'].unique():

	split_df = df_basin.get_group(basin_name)

	split_df.to_csv(os.path.join(OUTDIR, 'Argo_with_basins_%s.csv'%(basin_name)))

# But store also the complete one
df_out.to_csv(os.path.join(OUTDIR, '../CSV/Argo_with_basins.csv'))



