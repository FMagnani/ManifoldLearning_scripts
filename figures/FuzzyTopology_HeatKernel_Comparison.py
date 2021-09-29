#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 23:49:13 2021

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
from sklearn.manifold import spectral_embedding
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
Comparison between weights distributions and laplacian embedding of the Heat
Kernel and the Fuzzy Topological analysis.
"""

plt.style.use('seaborn')

knn = [5, 10, 15]

n_rows = len(knn)

fig, ax = plt.subplots(n_rows,3, gridspec_kw={'width_ratios':[2,1,1]})


for row in range(n_rows):
    
    n_neighbors = knn[row]
  
    # Knn graph
    knn_graph = kneighbors_graph(X, n_neighbors, mode='distance',
                                              metric='euclidean')
    knn_dists = sparse.find(knn_graph)[2]    

    # Heat kernel graph, t=5
    heatker_graph = compute_weights(knn_graph, 5)
    heatker_graph = (heatker_graph + heatker_graph.transpose())*.5
    heatker_weights = sparse.find(heatker_graph)[2]

    # Fuzzy topological graph
    fuztop_graph,_,_ = umap.umap_.fuzzy_simplicial_set(X, n_neighbors, 
                                                        123456, 'euclidean')
    fuztop_graph = (fuztop_graph + fuztop_graph.transpose())*.5
    fuztop_weights = sparse.find(fuztop_graph)[2]

    # First column
    # Careful! Weights of value 1 are removed from the fuzzy topological
    # counts. That's for better visualization.
    ax[row,0].set_ylabel("Neighbors: "+str(n_neighbors))
    heatker_label = "Heat kernel, Mean: "+str(heatker_weights.mean())[:4]
    fuztop_label = "Fuzzy top., Mean: "+str(fuztop_weights.mean())[:4]    
    ax[row,0].hist(heatker_weights, bins=50, color='b', alpha=.8, label=heatker_label)
    # Remove ones 
    ax[row,0].hist(fuztop_weights[ fuztop_weights<1 ], bins=50, color='r', alpha=.6, label=fuztop_label)
    ax[row,0].legend()
    ax[row,0].get_yaxis().set_ticks([])
    
    print('1')
    
    heatker_emb = spectral_embedding(heatker_graph, n_components=2)
    hk_x = heatker_emb[:,0]
    hk_y = heatker_emb[:,1]
    fuztop_emb = spectral_embedding(fuztop_graph, n_components=2)
    ft_x = fuztop_emb[:,0]
    ft_y = fuztop_emb[:,1]

    #Second column
    ax[row,1].scatter(hk_x,hk_y, c=labels, cmap='viridis', marker='+', s=25)
    ax[row,1].get_xaxis().set_ticks([])
    ax[row,1].get_yaxis().set_ticks([])
    
    print('2')
    
    #Third column
    ax[row,2].scatter(ft_x,ft_y, c=labels, cmap='viridis', marker='+', s=25)
    ax[row,2].get_xaxis().set_ticks([])
    ax[row,2].get_yaxis().set_ticks([])
    
    print('3\n')

ax[0,0].set_title("Weights distributions")
ax[0,1].set_title("Laplacian embedding\nof heat kernel graph")
ax[0,2].set_title("Laplacian embedding\nof fuzzy topological graph")

plt.show()

