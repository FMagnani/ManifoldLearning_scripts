#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 19:20:23 2021

@author: fede
"""

from utils import swiss_roll, compute_heatker_weights
import matplotlib.pyplot as plt
import umap
import scipy.sparse as sparse
import numpy as np
from sklearn.manifold import spectral_embedding

#%%

n_samples = 2000
noise = 0.0
seed = 123456

xscale = 1
yscale = 100
zscale = 1

data = swiss_roll(n_samples, noise, seed, 
                  xscale=xscale, yscale=yscale, zscale=zscale)
X = data.data
colors = data.t

#%%

n_neighbors_list = [5, 10, 20]

fig, ax = plt.subplots(3,3)

fig.suptitle("'Swiss roll' dataset embedded with different algorithms")

# j along columns
for j in range(3):
    
    n_neighbors = n_neighbors_list[j]
    
    # KNN distance-weighted graph
    knn_ind, knn_vals, _ = umap.umap_.nearest_neighbors(X, n_neighbors, 
                                                        metric='euclidean',
                                                        metric_kwds = {}, 
                                                        angular=False,
                                                        random_state=123456)
    
    # Getting the arrays to initialize sparse matrix in coordinate format
    coo_vals = knn_vals.reshape(n_samples*n_neighbors)
    coo_i = []
    for k in range(n_samples):
        coo_i.append( n_neighbors*[k] )
   
    coo_i = np.array(coo_i).reshape(n_samples*n_neighbors)
    coo_j = knn_ind.reshape(n_samples*n_neighbors)
    
    # knn graph as a COO sparse matrix
    knn_graph = sparse.coo_matrix((coo_vals, (coo_i, coo_j)))
    
    # fuzzy topological graph
    fuztop_graph, _, _ = umap.umap_.fuzzy_simplicial_set(X, n_neighbors=n_neighbors,
                                                    metric='euclidean',
                                                    random_state=123456)
    
    # heat kernel graph
    heatker_graph = compute_heatker_weights(knn_graph, t=5)
    
    # Laplacian embedding
    fuztop_emb = spectral_embedding(fuztop_graph, n_components=2)
    heatker_emb = spectral_embedding(heatker_graph, n_components=2)
    
    # Umap optimization
    umap_model = umap.UMAP(n_neighbors, 2, 'euclidean')
    umap_emb = umap_model.fit_transform(X)

    # PLOTTING
    for i in range(3):
        ax[i,j].get_xaxis().set_ticks([])
        ax[i,j].get_yaxis().set_ticks([])
    
    ax[2,j].set_xlabel("N_neighbors: "+str(n_neighbors))
    
    ax[0,j].scatter(heatker_emb[:,0], heatker_emb[:,1],
                    c = colors, cmap='viridis', marker='+', s=28)
    ax[1,j].scatter(fuztop_emb[:,0], fuztop_emb[:,1],
                    c = colors, cmap='viridis', marker='+', s=28)
    ax[2,j].scatter(umap_emb[:,0], umap_emb[:,1],
                    c = colors, cmap='viridis', marker='+', s=28)
    
    print("Made column")
    
ax[0,0].set_ylabel("Heat kernel (t=5)")
ax[1,0].set_ylabel("Fuzzy topology")
ax[2,0].set_ylabel("UMAP")

plt.show()






