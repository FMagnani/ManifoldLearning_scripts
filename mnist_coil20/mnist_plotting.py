#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 15:31:59 2021

@author: fede
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # For custom legend
from matplotlib.colors import Normalize # For custom legend
import pandas as pd
import numpy as np


path = "data/mnist_embedded/"
embedding_df = pd.read_csv(path+"mnist_embeddings.csv", index_col=0)
weights_df = pd.read_csv(path+"mnist_weights.csv", index_col=0)
dists_df = pd.read_csv(path+"mnist_distances.csv", index_col=0)
sigmas_df = pd.read_csv(path+"mnist_sigmas.csv", index_col=0)

labels = embedding_df["Labels"]

#%%
# Tot samples = 60000
n_samples_test = 60000
embedding_df = embedding_df[:][:n_samples_test]
labels = labels[:n_samples_test]

#%%

fig, axes = plt.subplots(3, 3)

neighbors_list = [5, 80, 300]

# j along columns
for j in [0,1,2]:

  ax = axes[:,j]

  n_neighbors = neighbors_list[j]

  dists = dists_df["Distances_"+str(n_neighbors)]
  weights = weights_df["Weights_"+str(n_neighbors)]

  sigmas = sigmas_df["Sigmas_"+str(n_neighbors)]

  annotation = "Weigths\nmean:"+str(weights.mean())[:4] +\
               "\nStd:"+str(np.std(weights))[:4]
  
  x_label =    "Sigma:"+str(sigmas.mean())[:5] +\
                "\xB1"+str(np.std(sigmas))[:5]


  fuztop_emb_x = embedding_df["FuzTop_x_"+str(n_neighbors)]
  fuztop_emb_y = embedding_df["FuzTop_y_"+str(n_neighbors)]
  umap_emb_x = embedding_df["Umap_x_"+str(n_neighbors)]
  umap_emb_y = embedding_df["Umap_y_"+str(n_neighbors)]
                               

  ax[0].hist(weights, bins=100, color='black')
  ax[1].scatter(fuztop_emb_x, fuztop_emb_y,
                c = labels, 
                cmap='tab10', marker='+', s=5)
  ax[2].scatter(umap_emb_x, umap_emb_y,
                c = labels, 
                cmap='tab10', marker='+', s=5)

  for i in range(3):
    ax[i].get_xaxis().set_ticks([])
    ax[i].get_yaxis().set_ticks([])

  ax[0].set_title("N neighbors = "+str(n_neighbors))
  ax[0].annotate(annotation, (0.5,0.6), xycoords='axes fraction')
  ax[0].set_xlabel(x_label)

  print(j)

# Legend
cmap = plt.cm.tab10
norm = Normalize(vmin=0, vmax=9)
legend_list = []
for c in range(10):
  color = cmap(norm(c))
  patch = mpatches.Patch(color=color, label=str(c))
  legend_list.append(patch)

axes[2,1].legend(handles=legend_list, 
                 loc='upper center', bbox_to_anchor=(0.5, 1.24),
                 ncol=10, fancybox=True, shadow=True)

axes[0,0].set_ylabel("Weights\ndistribution")
axes[1,0].set_ylabel("Fuzzy topological\nembedding")
axes[2,0].set_ylabel("UMAP\nembedding")

plt.show()

