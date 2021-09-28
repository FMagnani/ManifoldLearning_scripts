#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:21:02 2021

@author: fede
"""

from utils import swiss_roll
import matplotlib.pyplot as plt
from time import time
from basic_LaplacianEigenmap import basic_LapEig

scale = 1
data = swiss_roll(600, 0, 123456, scale)

X = data.data
labels = data.t

#%%

"""
TEST 1

My basic Laplacian Embedding for a variety of times
"""
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
    ax[i,j] = emb.plot_2D(ax[i,j])

stop = time()
print("time to run: "+str(stop-start))

plt.show()        


#%%

"""
TEST 2

My basic Laplacian Embedding for a variety of n_neighbors
"""
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
    ax[i,j] = emb.plot_2D(ax[i,j])

stop = time()
print("time to run: "+str(stop-start))

plt.show()        

