#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:25:59 2021

@author: fede
"""


from utils import swiss_roll
import matplotlib.pyplot as plt
from time import time
from basic_LaplacianEigenmap import basic_LapEig
from sklearn.manifold import SpectralEmbedding as SE


data = swiss_roll(600, 0, 123456)

X = data.data
labels = data.t

#%%

"""
TEST

Comparison between my basic Laplacian Embedding and the sklearn's 
SpectralEmbedding, for a variety of n_neighbors
"""
# basic_LapEig VS sklearn.manifold.SpectralEmbedding

# N_Neighbors change

fig, ax = plt.subplots(3,2)

fig.suptitle("My implementation VS sklearn implementation")

knn = [5, 15, 25]
ax_index_mine = [(0,0),(1,0),(2,0)]
ax_index_skl = [(0,1),(1,1),(2,1)]

my_time = 100

start = time()

for (n_neighbors,axis) in zip(knn, ax_index_mine):
    
    emb = basic_LapEig(data.data, data.t, n_neighbors, 2, 2500)
    
    i = axis[0]
    j = axis[1]
    ax[i,j] = emb.plot_2D(ax[i,j])

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

plt.show()        






