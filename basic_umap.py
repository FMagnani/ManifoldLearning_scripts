#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 11:37:19 2021

@author: fede
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import spectral_embedding

#%%

class basic_umap():
    """
    Class implementing a basic version of umap algorithm, without the low
    dimensional optimization. 
    The weighted graph of the probabilities is computed and then embedded
    in low dimensions through Laplacian Eigenmap.
    
    Pars:
        X: high dimensional data, shape (n_samples, n_features)
        labels: true labeling of the data, shape (n_samples)
        n_neighbors: hyperparameter for umap
        n_components: low dimensionality to which map the data
    """
    
    def __init__(self, X, labels, n_neighbors, n_components):
        
        self.X = X
        self.labels = labels
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        
        # X -> Pairwise Euclidean Distances, rho
        self.dist = np.square(euclidean_distances(X,X))
        n_points = self.dist.shape[0]
        self.rho = [ sorted(self.dist[i])[1] for i in range(n_points) ]
        
        # Pairwise Distances -> Pairwise Probabilities
        self.weighted_graph = np.zeros((n_points, n_points))
        self.sigma = []
        
        for i in range(n_points):
            
            local_k_sigma = self.k_sigma_i(i)
            
            local_sigma = self.sigma_binary_search(local_k_sigma, self.n_neighbors)
            self.sigma.append(local_sigma)
            self.weighted_graph[i] = self.weights_i(local_sigma, i)
        
        # Weighted graph symmetrization
        self.weighted_graph = (self.weighted_graph + self.weighted_graph.transpose())*0.5
        
        # Weighted graph -> Laplacian embedding
        self.embedding = spectral_embedding(self.weighted_graph, 
                                            n_components=self.n_components)
        
        
        
    def weights_i(self, sigma_i, i):
        """
        From the matrix of the distances, compute the set of probabilities 
        from the point i to all the others, given sigma_i.
        
        Pars:
            sigma: a value for sigma 
            i: index of the point
        Returns:
            weights: the set of probabilities from point 'i' to all the others        
        """    
        d = self.dist[i] - self.rho[i]
        d[ d<0 ] = 0
            
        # np.array = np.exp( np.array / float )
        weights = np.exp(-d/sigma_i)
    
        return weights


    def k_i(self, probs_i):
        """
        Given the set of the probabilities associated to point i, returns the
        nearest neighbors value for that point.
        
        Pars:
            probs: set of probabilites from point i to all the others.
        Returns:
            k: scalar
        
        """
        k = np.power(2, np.sum(probs_i))
        return k
    
    
    def sigma_binary_search(self, k_sigma, k_target):
        """
        Given a function k(sigma) and a scalar target k, find the sigma such
        that k_sigma(sigma) = k_target
        
        Pars:
            k_sigma: funcion taking sigma and returning k
            k: scalar, the k to be obtained
            
        Returns:
            sigma: scalar, such that k_sigma(sigma) = k_target
        """
        sigma_lower = 0
        sigma_upper = 1000
        for i in range(20):
            
            sigma_guess = (sigma_lower+sigma_upper)*0.5
            
            if (k_sigma(sigma_guess) < k_target):
                sigma_lower = sigma_guess
            else:
                sigma_upper = sigma_guess
            
            if (np.abs(k_sigma(sigma_guess) - k_target) <= 1e-5):
                break
            
        return sigma_guess
        
        
    def k_sigma_i(self, i):
        """
        Returns the function k(sigma) defined for a given point i.
        
        Pars:
            i: int, the index of the point considered
        Returns: 
            function: function(sigma) = k using the set set of probabilities
                      associated to the point i.
        """
        
        def function(sigma):
            
            k = self.k_i( self.weights_i(sigma, i) )
        
            return k
        
        return function

    
    def plot(self, ax):
        
        x = self.embedding[:,0]
        y = self.embedding[:,1]
                        
        pars_legend = 'n_neighbors: '+str(self.n_neighbors)
            
        ax.annotate(pars_legend, xy=(0.75,0.05),xycoords='axes fraction')
            
        ax.scatter(x,y, c=self.labels)
            
        return ax



#%%

# SCRIPT

from utils import swiss_roll

data = swiss_roll(600, 0, 123456, 1)

X = data.data
labels = data.t

emb = basic_umap(X, labels, 15, 2)

fig, ax = plt.subplots(1,1)

ax = emb.plot(ax)

plt.show()















