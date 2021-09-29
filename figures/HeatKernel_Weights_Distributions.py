#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 23:51:41 2021

@author: fede
"""

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
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
Heat kernel weighted graph has a parameter 't'. For various t choices, the 
resulting weights distribution is plotted.
But the heat kernel is applied to a knn_graph that has a 'n_neighbors' 
parameter. Various choices are made for it too.
INFO:
    Higher the 't' parameter, higer the resulting weights (pushed to 1).
    Smaller the 't' parameter, smaller the resulting weights (pushed to 0).
    Higher the 'n_neighbors' -> Higer the distances in average -> smaller the
    weights.
    Et viceversa.
    In few words, n_neighbors and t are opposite in their effect on the weights.
    High n_neighbors = Small t -> Weights to 0
    Small n_neighbors = High t -> Weights to 1
"""

"""
SYMMETRIZATION

Symmetrization has big effects on the weights distributions.
Comparison can be plotted in the two cases, just setting the global variable
'symmetrization' to True or False.
"""

# Heat Kernel weights distributions for various t and n_neighbors

# SYMMETRIZATION
symmetrization = True


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
        if symmetrization:
            heatker_graph = (heatker_graph + heatker_graph.transpose())*.5
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

plt.show()


