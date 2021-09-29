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

"""
The same function employed in 'basic_LapEig' class.
From knn_neighbors matrix weighted by the distance, it returns the matrix
weighted by Heat Kernel with time 't' in coo format.
"""

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

"""
STEP 1: knn matrix weighted by distance
Comparison between the sklearn implementation and the umap implementation:
they're equivalent.
INFO:
  The point itself is the first nn for umap. For sklearn not by default, 
                                              but the option is present.
  The weight of not connected points is 0 for both the implementations.    
"""


n_neighbors = 15

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

"""
From the knn-distances graph are computed: 
    the heat kernel weighted graph.
    the fuzzy topological graph.
The distributions of their weights are compared.
"""

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
ax.hist(fuztop_weights, bins=50, alpha=.6, color='r', label=fuztop_weights_label)

ax.legend()

plt.show()


#%%

"""
Heat kernel weighted graph has a parameter 't'. For various t choices, the 
resulting weights distribution is plotted.
INFO:
    Higher the 't' parameter, higer the resulting weights (pushed to 1).
    Smaller the 't' parameter, smaller the resulting weights (pushed to 0).
"""

# Heat Kernel weights distributions for various t

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

"""
Fuzzy topological graph: symmetrization
"""

# Fuzzy topological graph
fuztop_graph,_,_ = umap.umap_.fuzzy_simplicial_set(X, n_neighbors, 
                                                        123456, 'euclidean')
fuztop_weights = sparse.find(fuztop_graph)[2]

# Symmetrization average
fuztop_graph_1 = (fuztop_graph + fuztop_graph.transpose())*.5
fuztop_weights_1 = sparse.find(fuztop_graph_1)[2]

# Symmetrization fuzzy union
fuztop_graph_2 = fuztop_graph + fuztop_graph.transpose() + fuztop_graph.multiply(fuztop_graph)
fuztop_weights_2 = sparse.find(fuztop_graph_2)[2]

fig, ax = plt.subplots(3,1)

ax[0].hist(fuztop_weights, bins=50)
ax[1].hist(fuztop_weights_1, bins=50)
ax[2].hist(fuztop_weights_2, bins=50)

plt.show()


#%%

"""
Heat kernel graph: symmetrization
"""

# Heat kernel, a couple of t

# t=5
hk_1 = compute_weights(knn_graph, 5)
hk_w_1 = sparse.find(hk_1)[2]
# symmetrized
hk_symm_1 = (hk_1 + hk_1.transpose())*.5
hk_w_symm_1 = sparse.find(hk_symm_1)[2]

# t=100
hk_2 = compute_weights(knn_graph, 100)
hk_w_2 = sparse.find(hk_2)[2]
# symmetrized
hk_symm_2 = (hk_2 + hk_2.transpose())*.5
hk_w_symm_2 = sparse.find(hk_symm_2)[2]


# plot
fig, ax = plt.subplots(2,1)

ax[0].hist(hk_w_1, bins=50, color='b', alpha=.8)
ax[0].hist(hk_w_2, bins=50, color='r', alpha=.8)
ax[1].hist(hk_w_symm_1, bins=50, color='b', alpha=.8)
ax[1].hist(hk_w_symm_2, bins=50, color='r', alpha=.8)

plt.show()

