#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 19:32:18 2021

@author: FMagnani
GitHub: https://github.com/FMagnani
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#%%

path = "data/coil20_embedded/"
data = pd.read_csv(path+"coil20_embeddings.csv", index_col=0)
labels = data["Labels"]

#%%
"""
Figure 1: Laplacian Embedding of the Fuzzy Topological graph for euclidean
and cosine distances, for some n_neighbors values.
"""

fig, ax = plt.subplots(2,4)

for j in range(4):
    for i in range(2):
        ax[i,j].get_xaxis().set_ticks([])
        ax[i,j].get_yaxis().set_ticks([])

ax[0,0].set_ylabel("Euclidean distance")
ax[1,0].set_ylabel("Cosine distance")
    
fig.suptitle("Laplacian embedding of the fuzzy topological\ngraph of the coil20 dataset")

n_neighbors_list = [5,15,50,200]
# Rows
for (i, metric) in [ (0,'euclidean'), (1,'cosine') ]:
    # Columns
    for j in range(4):
        n_neighbors = n_neighbors_list[j]
        
        ax[0,j].set_xlabel("N_neighbors = "+str(n_neighbors))        
        
        x = data["FuzTop_x_"+str(metric)+"_"+str(n_neighbors)]
        y = data["FuzTop_y_"+str(metric)+"_"+str(n_neighbors)]
        
        ax[i,j].scatter(x, y, c=labels, cmap='nipy_spectral', marker='.')


plt.plot()

#%%

"""
Figure 2: Umap embedding of coil20, for euclidean and cosine distances,
for some n_neighbors values.
"""

fig, ax = plt.subplots(2,4)

for j in range(4):
    for i in range(2):
        ax[i,j].get_xaxis().set_ticks([])
        ax[i,j].get_yaxis().set_ticks([])

ax[0,0].set_ylabel("Euclidean distance")
ax[1,0].set_ylabel("Cosine distance")
    
fig.suptitle("Umap embedding of the coil20 dataset")

n_neighbors_list = [5,15,50,200]
# Rows
for (i, metric) in [ (0,'euclidean'), (1,'cosine') ]:
    # Columns
    for j in range(4):
        n_neighbors = n_neighbors_list[j]
        
        ax[0,j].set_xlabel("N_neighbors = "+str(n_neighbors))        
        
        x = data["Umap_x_"+str(metric)+"_"+str(n_neighbors)]
        y = data["Umap_y_"+str(metric)+"_"+str(n_neighbors)]
        
        ax[i,j].scatter(x, y, c=labels, cmap='nipy_spectral', marker='.')


plt.plot()

#%%

"""
Figures 3, 4: Embedding on the cylinder of the coil20 dataset.
"""

theta = data["Cylinder_theta"]
z = data["Cylinder_z"]

r = 1 # Radius of the cylinder

# Coordinate change for visualization
x = r * np.cos(theta)
y = r * np.sin(theta)
# z = z

u = np.arctan2(y,x)
v = z

fig1 = plt.figure()
ax1 = fig1.add_subplot(projection = '3d')
ax1.scatter(x,y,z, c = labels, cmap = 'nipy_spectral')
ax1.set_title("Umap embedding of coil20 dataset on the cylinder")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

fig2, ax2 = plt.subplots(1,1)
ax2.scatter(u, v, c=labels, cmap='nipy_spectral', s=20)
ax2.set_title("Umap embedding of coil20 dataset on the flattened cylinder")

plt.show()


























