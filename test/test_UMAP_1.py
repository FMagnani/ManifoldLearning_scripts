#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:49:43 2021

@author: fede
"""

from utils import swiss_roll
import umap
import umap.plot
from basic_umap import basic_umap
import matplotlib.pyplot as plt


data = swiss_roll(600, 0, 123456)

X = data.data
labels = data.t


#%%

"""
TEST - basic_umap

Test to visually check my basic umap implementation on the swiss_roll.
"""

emb = basic_umap(X, labels, 15, 2)

fig, ax = plt.subplots(1,1)
ax = emb.plot(ax)
plt.show()


#%%

"""
TEST - basic_umap VS real_umap

My basic umap implementation VS the real umap implementation.
Recall that the real one employs random noise.
"""


# Real umap embedding
real_umap = umap.UMAP(15, 2, n_epochs=11, learning_rate=0).fit(X)
# umap.plot.points(real_umap, labels=labels, 
#                   color_key_cmap='viridis', show_legend=False, )
real_emb = real_umap.embedding_


# My umap embedding
my_emb = basic_umap(X, labels, 15, 2)


# Plotting (just one n_neighbor case)
fig, ax = plt.subplots(1,2)

ax[0].set_title("Real umap embedding")
ax[0].scatter( -real_emb[:,0], -real_emb[:,1], c = labels )

ax[1].set_title("My umap embedding")
ax[1] = my_emb.plot(ax[1])

plt.show()





