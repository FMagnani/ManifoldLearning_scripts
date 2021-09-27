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

class basic_LapEig():
    
    def __init__(self, X, n_neighbors, n_components, t):
        # self.X = X
        # self.n_neighbors = n_neighbors
        self.n_components = n_components
        # self.t = t
        
        # Data -> knn graph (sparse matrix)
        knn_graph = kneighbors_graph(X, n_neighbors, mode='distance',
                                     metric='euclidean')
        
        # knn graph -> Weighted (symm) graph
        weighted_graph = self.compute_weights(knn_graph, t)
        
        # Weighted graph -> Laplacian Embedding
        self.laplacian_embedding = spectral_embedding(
                                                 weighted_graph, 
                                                 n_components=n_components
                                                 )
        
    
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


    def plot(self, dims=2):
        
        # 2D plot
        if (dims==2):
            
            fig, ax = plt.subplots(1,1)
            
            x = self.laplacian_embedding[:,0]
            y = self.laplacian_embedding[:,1]
            
            ax.scatter(x,y)
            
                    
            
            
            
            
            
            
            





