#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:07:45 2021

@author: fede
"""

import umap
from utils import swiss_roll
#from time import time
from sklearn.manifold import spectral_embedding
import matplotlib.pyplot as plt

n_samples = 200
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
TEST

Just a simple embedding
"""

umap_seed = 1533462
n_components = 2
n_neighbors = 15

w_graph, sigmas, rho = umap.umap_.fuzzy_simplicial_set(X, n_neighbors, 
                                                umap_seed, 'euclidean')

emb = spectral_embedding(w_graph, n_components=n_components)

# Plotting

fig, ax = plt.subplots(1,1)

ax.set_title("Fuzzy topological embeddings of swiss roll dataset")

x = emb[:,0]
y = emb[:,1]

ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

pars = "N_neighbors: "+str(n_neighbors)
ax.set_xlabel(pars)

ax.scatter(x,y, c = labels, cmap='viridis')

plt.show()

#%%

"""
TEST

How do the Fuzzy Topological embedding changes with 'n_neighbors' parameter?
"""

fig, ax = plt.subplots(2,2)

fig.suptitle("Fuzzy topological embeddings of swiss roll dataset")

knn = [5, 15, 25, 30]
ax_index = [(0,0),(0,1),(1,0),(1,1)]

umap_seed = 1533462
n_components = 2

for (knn,axis) in zip(knn, ax_index):
    
    n_neighbors = knn
    ax_ij = ax[axis]
    
    w_graph, sigmas, rho = umap.umap_.fuzzy_simplicial_set(X, n_neighbors, 
                                                           umap_seed, 
                                                           'euclidean')

    emb = spectral_embedding(w_graph, n_components=n_components)

    ax_ij.get_xaxis().set_ticks([])
    ax_ij.get_yaxis().set_ticks([])

    pars = "N_neighbors: "+str(n_neighbors)
    ax_ij.set_xlabel(pars)

    ax_ij.scatter(x,y, c = labels, cmap='viridis')

plt.show()        

#%%

"""
TEST

How do the Fuzzy Topological embedding changes with the scale and density
of the dataset?
"""

yscales = [1, 10, 100]
n_samples = [200, 2000, 5000]

fig, ax = plt.subplots(3,3)

fig.suptitle("Fuzzy topological representations of the 'swiss roll' dataset")

for i in range(3):
    for j in range(3):
        
        ysc = yscales[i]
        n = n_samples[j]
        
        data = swiss_roll(n, 0, 123456, xscale=1,yscale=ysc,zscale=1)
        X = data.data
        labels = data.t
        ax_ij = ax[i,j]
        
        w_graph, sigmas,rho = umap.umap_.fuzzy_simplicial_set(X, 15, 
                                                              umap_seed, 
                                                              'euclidean')

        emb = spectral_embedding(w_graph, n_components=2)
        
        x = emb[:,0]
        y = emb[:,1]

        pars = "Samples: "+str(n)+", y scale: "+str(ysc)
        ax_ij.set_xlabel(pars)
        
        ax_ij.get_xaxis().set_ticks([])
        ax_ij.get_yaxis().set_ticks([])

        ax_ij.scatter(x,y, c=labels, cmap='viridis', marker='+', s=25)

        print('xd')

plt.show()


















