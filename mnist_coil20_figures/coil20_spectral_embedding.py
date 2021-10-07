#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 17:30:18 2021

@author: FMagnani
GitHub: https://github.com/FMagnani
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.patches as mpatches # For custom legend
from matplotlib.colors import Normalize # For custom legend

#%%

path = "data/alg_comparison/"
emb = pd.read_csv(path+"coil20_full_embedding.csv", index_col=0)
labels = emb["Labels"]

#%%

n_neighbors_list = [5, 10, 15]

fig, axes = plt.subplots(2,3)
plt.rcParams.update(plt.rcParamsDefault)

fig.set_facecolor('white')

# Along columns
for j in [0,1,2]:

  n_neighbors = n_neighbors_list[j]
  ax = axes[:,j]
  
  facecol = 'white'
  ax[0].set_facecolor(facecol)
  ax[1].set_facecolor(facecol)
    
  ax[0].scatter(emb["LapEig_x_"+str(n_neighbors)], emb["LapEig_y_"+str(n_neighbors)], 
                c=emb["Labels"], marker='o', s=15, cmap="nipy_spectral")
  ax[1].scatter(emb["FuzTop_x_"+str(n_neighbors)], emb["FuzTop_y_"+str(n_neighbors)], 
                c=emb["Labels"], marker='o', s=15, cmap="nipy_spectral")

  ax[0].get_xaxis().set_ticks([])
  ax[0].get_yaxis().set_ticks([])
  ax[1].get_xaxis().set_ticks([])
  ax[1].get_yaxis().set_ticks([])

  ax[1].set_xlabel("N neighbors: "+str(n_neighbors), fontsize=14)

# Legend
cmap = plt.cm.nipy_spectral
norm = Normalize(vmin=0, vmax=labels.max())
legend_list = []
n_legend = len(labels.unique())
for c in range(n_legend+1)[1:]: # I want to start from 1
  color = cmap(norm(c))
  patch = mpatches.Patch(color=color, label=str(c))
  legend_list.append(patch)

axes[0,1].legend(handles=legend_list, 
                 loc='lower center', bbox_to_anchor=(0.3, -0.22),
                 ncol=10)

axes[0,0].set_ylabel("Laplacian Eigenmap", fontsize=14)
axes[1,0].set_ylabel("Fuzzy topological\nspectral embedding", fontsize=14)

fig.show()

