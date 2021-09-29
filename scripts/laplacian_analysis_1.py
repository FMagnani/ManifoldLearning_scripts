#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:16:51 2021

@author: fede
"""

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
# Laplacian Eigenmap
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sparse
# Fuzzy Topological
import umap
# Spectral Embedding
# from sklearn.manifold import spectral_embedding
# Data
from utils import swiss_roll
#from time import time

n_samples = 2000
noise = 0.0
seed = 123456

xsc = 1
ysc = 100
zsc = 1

data = swiss_roll(n_samples, noise, seed, xsc,ysc,zsc)

X = data.data
labels = data.t

#%%

def compute_weights(knn_graph, t):
        
    inv_t = 1/t
        
    rows = knn_graph.nonzero()[0]
    cols = knn_graph.nonzero()[1]
        
    entries = []
        
    for i in range(len(rows)):
            
        dist = knn_graph[ rows[i], cols[i] ]
        weight = np.exp( -inv_t*np.power(dist, 2) )
        entries.append(weight)
            
    weighted_graph = sparse.coo_matrix((entries, (rows, cols)))
    weighted_graph = weighted_graph.tocsr()
        
    return weighted_graph



#%%

n_neighbors = 15

# STEP 1: knn matrix weighted by distance
# Comparison between the sklearn implementation and the umap implementation
# They're equivalent
# INFO:
#   The point itself is the first nn for umap. For sklearn not by default, but the option is present.
#   The weight of not connected points is 0 for both the implementations.    
#

# Distance matrix + knn neighbors
knn_graph = kneighbors_graph(X, n_neighbors, mode='distance',
                                             metric='euclidean')
knn_dists = sparse.find(knn_graph)[2]

# umap's distance matrix + knn neighbors
umap_dists = umap.umap_.nearest_neighbors(X, n_neighbors, metric='euclidean',
                                          metric_kwds = {}, angular=False,
                                          random_state=123456)[1]

umap_dists = umap_dists[:,1:] # Discard first nn, that's the point itself
umap_dists = umap_dists.reshape(n_samples*(n_neighbors-1)) # Reshape

# STEP 1 - PLOTTING

fig, ax = plt.subplots(1,1)
plt.style.use('seaborn')

fig.suptitle("Eucledian distances + knn neighbors \n ~ umap vs sklearn implementation ~")

sklearn_dists_label = "Sklearn, ("+str(knn_dists.shape[0])+" entries)"
umap_dists_label = "Umap, ("+str(umap_dists.shape[0])+" entries)"

ax.hist(knn_dists, bins=50, alpha=.8, color='b', label=sklearn_dists_label)
ax.hist(umap_dists, bins=50, alpha=.6, color='r', label=umap_dists_label)

ax.legend()

plt.show()


#%%


# STEP 2 - Dist_KNN matrix to weighted graph

# Heat kernel applied to knn- depends on t
heatker_graph = compute_weights(knn_graph, 5)
heatker_weights = sparse.find(heatker_graph)[2]

# Fuzzy topological representation
fuztop_graph,_,_ = umap.umap_.fuzzy_simplicial_set(X, n_neighbors, 
                                                123456, 'euclidean')
fuztop_weights = sparse.find(fuztop_graph)[2]


# STEP 2 - PLOTTING

fig, ax = plt.subplots(1,1)
plt.style.use('seaborn')

fig.suptitle("Weighted graph \n ~ Fuzzy topological (umap) vs Heat Kernel ~")

heatker_weights_label = "Heat Kernel weights distribution (t=5)"
fuztop_weights_label = "Fuzzy topological weights distribution"

ax.hist(heatker_weights, bins=50, alpha=.8, color='b', label=heatker_weights_label)
ax.hist(fuztop_weights[fuztop_weights<1], bins=50, alpha=.6, color='r', label=fuztop_weights_label)

ax.legend()

plt.show()


#%%

# STEP 2 B - Heat Kernel weights distributions for various t

times = [5, 7, 20, 200]

fig, ax = plt.subplots(1,1)

for i in range(len(times)):
    
    t = times[i]
    
    heatker_graph = compute_weights(knn_graph, t)
    heatker_weights = sparse.find(heatker_graph)[2]

    ax.hist(heatker_weights, bins=50, alpha=.7, 
              label="t = "+str(t))

fig.suptitle("Heat kernel weights distributions for various t")
ax.legend()

plt.show()


#%%

# STEP 2 C - Heat Kernel weights distributions for various t and n_neighbors

plt.style.use('seaborn')

times = [5, 25, 100]
knn = [5, 10, 15]

n_rows = len(knn)

fig, ax = plt.subplots(n_rows,2, gridspec_kw={'width_ratios':[1,3]})

for row in range(n_rows):
    
    n_neighbors = knn[row]
    
    knn_graph = kneighbors_graph(X, n_neighbors, mode='distance',
                                              metric='euclidean')
    knn_dists = sparse.find(knn_graph)[2]    
    
    ax[row,0].set_ylabel("Neighbors: "+str(n_neighbors))
    ax[row,0].hist(knn_dists, bins=50, color='black')
    ax[row,0].annotate("Mean: "+str(knn_dists.mean())[:4],
                       (0.6,0.8), xycoords='axes fraction')
    
    axis = ax[row,1]    
    
    for t in times:
        
        heatker_graph = compute_weights(knn_graph, t)
        heatker_weights = sparse.find(heatker_graph)[2]

        label = "t = "+str(t)+", mean: "+str(heatker_weights.mean())[:4]

        axis.hist(heatker_weights, bins=50, alpha=.7, 
                label=label)
        axis.legend()

        print(row, t)
    

ax[0,1].get_xaxis().set_ticks([])
ax[1,1].get_xaxis().set_ticks([])

for axis in ax[:,0][:-1]:
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])
ax[2,0].get_yaxis().set_ticks([])

ax[0,1].set_title("Heat kernel weights distributions (various t)")
ax[0,0].set_title("Distances distribution (knn applied)")
#fig.suptitle("Heat kernel weights distributions for various t")


plt.show()




















