#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 22:59:30 2021

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
emb = pd.read_csv(path+"5digit_mnist_embedding.csv", index_col=0)
labels = data["Labels"]

#%%

n_neighbors_list = [8, 20, 50]

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
                c=emb["Labels"], marker='*', s=5, cmap="jet")
  ax[1].scatter(emb["FuzTop_x_"+str(n_neighbors)], emb["FuzTop_y_"+str(n_neighbors)], 
                c=emb["Labels"], marker='*', s=5, cmap="jet")

  ax[0].get_xaxis().set_ticks([])
  ax[0].get_yaxis().set_ticks([])
  ax[1].get_xaxis().set_ticks([])
  ax[1].get_yaxis().set_ticks([])

  ax[1].set_xlabel("N neighbors: "+str(n_neighbors))

# Legend
cmap = plt.cm.jet
norm = Normalize(vmin=0, vmax=4)
legend_list = []
for c in range(5):
  color = cmap(norm(c))
  patch = mpatches.Patch(color=color, label=str(c))
  legend_list.append(patch)

axes[0,1].legend(handles=legend_list, 
                 loc='lower center', bbox_to_anchor=(0.3, -0.2),
                 ncol=5)

axes[0,0].set_ylabel("Laplacian Eigenmap")
axes[1,0].set_ylabel("Fuzzy topological\nspectral embedding")

fig.show()

#%%



data[ data.columns[1:] ]*10000

fig, ax = plt.subplots(2,4)

for j in range(4):
    for i in range(2):
        ax[i,j].get_xaxis().set_ticks([])
        ax[i,j].get_yaxis().set_ticks([])

ax[0,0].set_ylabel("Laplacian Eigenmap")
ax[1,0].set_ylabel("Spectral embedding of the\nfuzzy topological graph")
    
fig.suptitle("Laplacian eigenmap vs spectral embedding\nof the fuzzy topological graph\nfor the mnist dataset")

n_neighbors_list = [5,15,50,200]
# Columns
for j in range(4):
    n_neighbors = n_neighbors_list[j]
        
    ax[0,j].set_xlabel("N_neighbors = "+str(n_neighbors))            
    
    LEx = data["LapEig_x_"+str(n_neighbors)]
    LEy = data["LapEig_y_"+str(n_neighbors)]
    
    FTx = data["FuzTop_x_"+str(n_neighbors)]
    FTy = data["FuzTop_y_"+str(n_neighbors)]

    ax[0,j].scatter(LEx, LEy, c=labels, cmap='nipy_spectral', marker='.')            
    ax[1,j].scatter(FTx, FTy, c=labels, cmap='nipy_spectral', marker='.')

    print("Made neighbors=", n_neighbors)

plt.show()

#%%


fig, ax = plt.subplots(1,1)

n_neighbors = 15

# LEx = data["LapEig_x_"+str(n_neighbors)]
# LEy = data["LapEig_y_"+str(n_neighbors)]

FTx = data["FuzTop_x_"+str(n_neighbors)]
FTy = data["FuzTop_y_"+str(n_neighbors)]

ax.scatter(FTx, FTy, c=labels, cmap='nipy_spectral', marker='.') 

plt.show()



