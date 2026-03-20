#!/usr/bin/env python

# Edited by amiro and eterzic 21.02.2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

import MapPlotter as mp


import Basins

from datetime import datetime



INPUTDIR     = '/Users/elena/Projects/IRB/MedClim/Data/Argo/Basins/'

OUTDIR       ='/Users/elena/Projects/IRB/MedClim/Plots/Argo/Basins/'

basin_list = [Basins.manca.df2, Basins.manca.df3, Basins.manca.df4, Basins.manca.ds2,Basins.manca.df1,
			  Basins.manca.ds1, Basins.manca.ds3, Basins.manca.ds4, Basins.manca.dt1,Basins.manca.dt2,
			  Basins.manca.dt3, Basins.manca.di1, Basins.manca.di3, Basins.manca.dj1, Basins.manca.dj2, 
			  Basins.manca.dj3, Basins.manca.dj6, Basins.manca.dj4, Basins.manca.dj7, Basins.manca.dj8, 
			  Basins.manca.dj5, Basins.manca.dh1, Basins.manca.dh2, Basins.manca.dh3, Basins.manca.dl1, 
			  Basins.manca.dl2, Basins.manca.dl3, Basins.manca.dl4]


# Instance MapPlotter class
plotter = mp.MapPlotter(projection='PlateCarree')

# Create basic parameters dictionary
params  = plotter.defaultParams()
# DPI
params['dpi']         = 100
# Which features need to be plotted?
params['features']    = ['coastline'] #'continents','rivers''tilemap'
params['map_zoom']    = 9
params['map_style']   = 'satellite'
# A bit of formatting on the title and axis
params['xlabel']      = ['Longitude (deg)']
params['ylabel']      = ['Latitude (deg)']
# A bit of formatting on the colorbar
params['cmap']        = 'jet'
params['orientation'] = 'vertical'
params['tick_format'] = 'datetime'
params['shrink']      = 0.38



for ibasin, basin in enumerate(basin_list):

	csv_file = os.path.join(INPUTDIR, 'Argo_with_basins_%s.csv'%(basin.abbrev)) 

	try:
		df           = pd.read_csv(csv_file, parse_dates=['TIME'])
	except FileNotFoundError:
		print(f"Skipping {csv_file}: not found.")
		continue
	except pd.errors.EmptyDataError:
		print(f"Skipping {csv_file}: empty or corrupted CSV.")
		continue

	if df.empty:
		print(f"Skipping {csv_file}: dataframe has no rows.")
		continue

	df['DATE']   = [datetime.timestamp(d) for d in df.TIME]

	print(basin.name)

	fig = plt.figure(figsize=(12,10))
	ax  = fig.add_subplot(1,1,1,projection=plotter.projection)
	params['fig'] = fig
	params['ax']  = ax

	#Crop time-wise
	start_date   = datetime(2001,1,1)
	end_date     = datetime(2025,12,31)

	df_aux       = df[df.TIME.between(start_date, end_date)]

	df_float = df_aux.sort_values(by='TIME', ascending=True)
	

	# Plot trajectories
	lat     = df_float['LATITUDE' ]
	lon     = df_float['LONGITUDE']
	time    = df_float['DATE']

	# Limits for the plot
	params['xlim']        = [np.nanmin(lon)*0.9, np.nanmax(lon)*1.1]
	params['ylim']        = [np.nanmin(lat)*0.9, np.nanmax(lat)*1.1]

	# Plot
	plotter.scatter(lon,lat,data=time,size=10, params=params)

	plotter.save(os.path.join(OUTDIR, '%s.png'%(basin.name.replace(' ', '_'))), dpi=300)

	plt.clf()
	plt.close()