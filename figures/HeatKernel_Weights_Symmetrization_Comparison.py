#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 23:44:04 2021

@author: fede
"""

import numpy as np
import matplotlib.pyplot as plt
# Laplacian Eigenmap
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sparse
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
Effect of symmetrization on the heat kernel weights distributions.
"""

plt.style.use('seaborn')

times = [5, 25, 100]
knn = [5, 10, 15]

n_rows = len(knn)

fig, ax = plt.subplots(n_rows,2, gridspec_kw={'width_ratios':[1,1]})

for row in range(n_rows):
    
    n_neighbors = knn[row]
    
    knn_graph = kneighbors_graph(X, n_neighbors, mode='distance',
                                              metric='euclidean')
    knn_dists = sparse.find(knn_graph)[2]    
    
    ax[row,0].set_ylabel("Neighbors: "+str(n_neighbors))
    
    for t in times:
        
        heatker_graph = compute_weights(knn_graph, t)
        heatker_weights = sparse.find(heatker_graph)[2]

        label = "t="+str(t)+", mean:"+str(heatker_weights.mean())[:4] +\
                " std: "+str(np.std(heatker_weights))[:4]

        ax[row,0].hist(heatker_weights, bins=50, alpha=.8, 
                label=label)
        ax[row,0].legend()

        ax[row,0].get_xaxis().set_ticks([])        
        ax[row,0].get_yaxis().set_ticks([])        

        # Symmetrization
        heatker_graph = (heatker_graph + heatker_graph.transpose())*.5
        heatker_weights = sparse.find(heatker_graph)[2]

        label = "t="+str(t)+", mean:"+str(heatker_weights.mean())[:4] +\
                " std: "+str(np.std(heatker_weights))[:4]

        ax[row,1].hist(heatker_weights, bins=50, alpha=.8, 
                label=label)
        ax[row,1].legend()
        
        ax[row,1].get_xaxis().set_ticks([])        
        ax[row,1].get_yaxis().set_ticks([])                

        print(row, t)
    
    
    
ax[0,0].set_title("Weights distributions\nfor directed graph")
ax[0,1].set_title("Weights distributions\nfor symmetrized graph")

plt.show()

