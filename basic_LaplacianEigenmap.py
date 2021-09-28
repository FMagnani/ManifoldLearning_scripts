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


    def plot_2D(self, ax):

        x = self.embedding[:,0]
        y = self.embedding[:,1]
                        
        pars_legend = 'n_neighbors: '+str(self.n_neighbors)+\
                        '\nt: '+str(self.t)   
        ax.annotate(pars_legend, xy=(0.75,0.05),xycoords='axes fraction')
            
        ax.scatter(x,y, c=self.labels)
            
        return ax
            
    def plot_3D(self):
            
        fig = plt.figure()
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
                        
            
        


























