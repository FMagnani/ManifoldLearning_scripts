#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:00:09 2021

@author: fede
"""

from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sparse
import numpy as np
from sklearn.manifold import spectral_embedding
import matplotlib.pyplot as plt

#%%

# OK FOR SWISS ROLL
# NOT YET READY FOR MNIST
class basic_LapEig():
    
    def __init__(self, X, labels, n_neighbors, n_components, t):
        self.X = X
        self.labels = labels
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.t = t
        
        # Data -> knn graph (sparse matrix)
        # knn graph is not symmetric.
        knn_graph = kneighbors_graph(X, n_neighbors, mode='distance',
                                     metric='euclidean')
                
        # knn graph -> Weighted graph
        weighted_graph = self.compute_weights(knn_graph, t)
        
        # Symmetrize weighted graph (sort of standard way to do this)
        weighted_graph = (weighted_graph + weighted_graph.transpose())*.5
        
        # Weighted graph -> Laplacian Embedding
        self.embedding = spectral_embedding(weighted_graph, 
                                            n_components=n_components)
        
    
    def compute_weights(self, knn_graph, t):
        
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


    def plot(self, ax, dims=2):
        
        # 2D plot
        if (dims==2):
            
            x = self.embedding[:,0]
            y = self.embedding[:,1]
                        
            pars_legend = 'n_neighbors: '+str(self.n_neighbors)+\
                            '\nt: '+str(self.t)   
            ax.annotate(pars_legend, xy=(0.75,0.05),xycoords='axes fraction')
            
            ax.scatter(x,y, c=self.labels)
            
            return ax
            
        #3D plot
        if (dims==3):
            
            ax = fig.add_subplot(projection = '3d')
        
            x = self.embedding[:,0]
            y = self.embedding[:,1]
            z = self.embedding[:,2]
        
            title = 'Laplacian Eigenmap embedding\nknn: '+\
                    str(self.n_neighbors)+', t: '+str(self.t)
        
            ax.set_title(title)
            ax.set_xlabel('First eigenvector')
            ax.set_ylabel('Second eigenvector')
            ax.set_zlabel('Third eigenvector')
            
            ax.scatter(x,y,z, c = self.labels, cmap = 'viridis')        
            
            return ax
            
            
            
#%%

# SCRIPTS

from utils import swiss_roll
from time import time
from sklearn.manifold import SpectralEmbedding as SE

scale = 5
data = swiss_roll(600, 0, 123456, scale)

#%%

# Time change

fig, ax = plt.subplots(2,2)

fig.suptitle("Laplacian Eigenmap embedding")

times = [5, 33, 66, 100]
ax_index = [(0,0),(0,1),(1,0),(1,1)]

start = time()
for (t,axis) in zip(times, ax_index):
    
    emb = basic_LapEig(data.data, data.t, 15, 2, t)
    
    i = axis[0]
    j = axis[1]
    ax[i,j] = emb.plot(ax[i,j], 2)

stop = time()
print("time to run: "+str(stop-start))

fig.show()        


#%%

# N_Neighbors change

fig, ax = plt.subplots(2,2)

fig.suptitle("Laplacian Eigenmap embedding")

knn = [5, 15, 20, 30]
ax_index = [(0,0),(0,1),(1,0),(1,1)]

start = time()
for (knn,axis) in zip(knn, ax_index):
    
    emb = basic_LapEig(data.data, data.t, knn, 2, 30)
    
    i = axis[0]
    j = axis[1]
    ax[i,j] = emb.plot(ax[i,j], 2)

stop = time()
print("time to run: "+str(stop-start))

fig.show()        

#%%

# basic_LapEig VS sklearn.manifold.SpectralEmbedding

# N_Neighbors change

fig, ax = plt.subplots(3,2)

fig.suptitle("My implementation VS sklearn implementation")

knn = [5, 15, 25]
ax_index_mine = [(0,0),(1,0),(2,0)]
ax_index_skl = [(0,1),(1,1),(2,1)]

my_time = scale*scale*100

start = time()

for (n_neighbors,axis) in zip(knn, ax_index_mine):
    
    emb = basic_LapEig(data.data, data.t, n_neighbors, 2, 2500)
    
    i = axis[0]
    j = axis[1]
    ax[i,j] = emb.plot(ax[i,j], 2)

for (n_neighbors,axis) in zip(knn, ax_index_skl):
    
    emb = SE(n_components=2, n_neighbors=n_neighbors).fit_transform(data.data)
    
    i = axis[0]
    j = axis[1]
    
    ax[i,j].scatter( emb[:,0], emb[:,1], c = data.t )
    
    pars_legend = 'n_neighbors: '+str(n_neighbors)+\
                            '\nt: inf'   
    ax[i,j].annotate(pars_legend, xy=(0.75,0.05),xycoords='axes fraction')
           


stop = time()
print("time to run: "+str(stop-start))

fig.show()        





























