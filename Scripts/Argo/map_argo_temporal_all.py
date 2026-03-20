#!/usr/bin/env python

# eterzic 11.06.2024 
# Use the data frame saved in the save_Argo_data.py code
# Plot a trajectory map with a temporal color bar


import MapPlotter as mp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from datetime import datetime




INPUTDIR = '/Users/elena/Projects/IRB/MedClim/Data/Argo/CSV/'

OUTDIR   = '/Users/elena/Projects/IRB/MedClim/Plots/Argo/'

df = pd.read_csv(os.path.join(INPUTDIR, 'Argo_with_basins.csv'), parse_dates=['TIME'])


df['DATE'] = [datetime.timestamp(d) for d in df.TIME]


# Instance MapPlotter class
plotter = mp.MapPlotter(projection='PlateCarree')

# Create basic parameters dictionary
params  = plotter.defaultParams()
# DPI
params['dpi']         = 100
# Limits for the plot
params['xlim']        = [-6, 36]
params['ylim']        = [30, 46]
# Which features need to be plotted?
params['features']    = ['coastline']#['coastline']#'continents','rivers'
params['map_zoom']    = 9
params['map_style']   = 'satellite'
# A bit of formatting on the title and axis
params['xlabel']      = ['Longitude (deg)']
params['ylabel']      = ['Latitude (deg)']
# A bit of formatting on the colorbar
params['cmap']        = 'jet'
params['orientation'] = 'vertical'
params['tick_format'] = 'datetime'
params['shrink']      = 0.6


fig = plt.figure(figsize=(12,10))
ax  = fig.add_subplot(1,1,1,projection=plotter.projection)
params['fig'] = fig
params['ax']  = ax


# Plot trajectories
lat     = df['LATITUDE' ]
lon     = df['LONGITUDE']
time    = df['DATE']

# Plot
plotter.scatter(lon,lat,data=time,size=10, params=params)

#ax.legend(df.PLATFORM_NUMBER.unique(), loc='best')

plotter.save(os.path.join(OUTDIR, 'Temporal_trajectory_all.png'), dpi=300)

plt.clf()
plt.close()

