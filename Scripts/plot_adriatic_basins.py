#!/usr/bin/env python

# Edited by amiro and eterzic 21.02.2021
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import Basins

# Load  basin
#basin = Basins.worldseas.kotor

north = Basins.adriatic.north
mid = Basins.adriatic.mid
south = Basins.adriatic.south

# Plot
plt.figure(1,(8,6),dpi=100)

plt.plot(north.x,north.y,'o-k')
plt.scatter(north.centroid.x,north.centroid.y,marker='x',c='k')

plt.plot(mid.x,mid.y,'o-r')
plt.scatter(mid.centroid.x,mid.centroid.y,marker='x',c='r')

plt.plot(south.x,south.y,'o-b')
plt.scatter(south.centroid.x,south.centroid.y,marker='x',c='b')

plt.show()