#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:18:17 2021

@author: fede
"""

from utils import swiss_roll
from basic_LaplacianEigenmap import basic_LapEig
import matplotlib.pyplot as plt

#%%

n_samples = 2000
noise = 0.0
seed = 123456

xscale = 1
yscale = 100
zscale = 1

data = swiss_roll(n_samples, noise, seed, 
                  xscale=xscale, yscale=yscale, zscale=zscale)
X = data.data
t_roll = data.t

#%%

times = [5, 25, 100]
n_neighbors = [5, 10, 15]

fig, ax = plt.subplots(3,3)

fig.suptitle("2-dim representations of the 'swiss roll' dataset")

for i in range(3):
    for j in range(3):
        
        t = times[i]
        n = n_neighbors[j]
        
        ax_ij = ax[i,j]
        
        emb = basic_LapEig(X, t, n, 2, t)
        
        x = emb.embedding[:,0]
        y = emb.embedding[:,1]

        pars = "N: "+str(n)+", t: "+str(t)
        ax_ij.set_xlabel(pars)
        
        ax_ij.get_xaxis().set_ticks([])
        ax_ij.get_yaxis().set_ticks([])

        ax_ij.scatter(x,y, c=t_roll, cmap='viridis', marker='+', s=25)

plt.show()









