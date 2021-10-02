#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:00:09 2021

@author: fede
"""

from sklearn.neighbors import kneighbors_graph
from umap.umap_ import nearest_neighbors
import scipy.sparse as sparse
import numpy as np
from sklearn.manifold import spectral_embedding
import matplotlib.pyplot as plt

def heatKernel_graph(X, n_neighbors, t, seed=123456):
    """
    Computes the heat kernel weighted graph, gicen parameters 'n_neighbors'
    and 't'. Returns the sparse adjacency matrix in csr format.
    PARS:
        X: numpy array of shape (n_samples, n_features)
        n_neighbors: int
        t: scalar - be careful since it depends on the distances distributions
    """
    
    n_samples = X.shape[0]
    
    # KNN distance-weighted graph
    knn_ind, knn_vals, _ = nearest_neighbors(X, n_neighbors, 
                                             metric='euclidean',
                                             metric_kwds = {}, 
                                             angular=False,
                                             random_state=seed)
    
    # Getting the arrays to initialize sparse matrix in coordinate format
    coo_vals = knn_vals.reshape(n_samples*n_neighbors)
    coo_i = []
    for k in range(n_samples):
        coo_i.append( n_neighbors*[k] )   
    coo_i = np.array(coo_i).reshape(n_samples*n_neighbors)
    coo_j = knn_ind.reshape(n_samples*n_neighbors)
    
    # knn graph as a COO sparse matrix
    knn_graph = sparse.coo_matrix((coo_vals, (coo_i, coo_j)))
   
    # heat kernel graph
    heatker_graph = compute_heatker_weights(knn_graph, t)
    
    return heatker_graph

    

def compute_heatker_weights(knn_graph, t):
    """
    Given a neighbors graph (eventually weighted by the distance), the 
    graph weighted by the heat kernel at time 't' is returned.
    
    PARS:
        knn_graph: Should be one of the classes of scipy.sparse
        t: time parameter for the heat kernel
    RETURNS:
        weighted_graph: The weighted graph, symmetrized by average,
                        in csr format.
    """
        
    inv_t = 1/t
        
    knn_graph = knn_graph.tocsr()
        
    rows = knn_graph.nonzero()[0]
    cols = knn_graph.nonzero()[1]
        
    entries = []
        
    for i in range(len(rows)):
            
        dist = knn_graph[ rows[i], cols[i] ]
        weight = np.exp( -inv_t*np.power(dist, 2) )
        entries.append(weight)
            
    weighted_graph = sparse.coo_matrix((entries, (rows, cols)))
    weighted_graph = weighted_graph.tocsr()
        
    weighted_graph = (weighted_graph + weighted_graph.transpose())*0.5
        
    return weighted_graph

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
        self.weighted_graph = compute_heatker_weights(knn_graph, t)
        
        # Symmetrize weighted graph (sort of standard way to do this)
        self.weighted_graph = (self.weighted_graph + self.weighted_graph.transpose())*.5
        
        # Weighted graph -> Laplacian Embedding
        self.embedding = spectral_embedding(self.weighted_graph, 
                                            n_components=n_components)
        


    def plot_2D(self, ax):

        x = self.embedding[:,0]
        y = self.embedding[:,1]
                        
        pars_legend = 'n_neighbors: '+str(self.n_neighbors)+\
                        ', t: '+str(self.t)   
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlabel(pars_legend)
            
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
                        
            
        


























