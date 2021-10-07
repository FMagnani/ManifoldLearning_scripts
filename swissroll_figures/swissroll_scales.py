#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:28:00 2021

@author: FMagnani
GitHub: https://github.com/FMagnani
"""

from utils import swiss_roll
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt

#%%

dataset_x = swiss_roll(2000, 0, 123, 10,10,1)
dataset_z = swiss_roll(2000, 0, 123, 1,10,10)

dataset_x.plot()
dataset_z.plot()

x = dataset_x.data
labels_x = dataset_x.t

z = dataset_z.data
labels_z = dataset_z.t

#%%

model = SpectralEmbedding(n_components=2, n_neighbors=10)

x1 = model.fit_transform(x)
z1 = model.fit_transform(z)

fig, ax = plt.subplots(2,1)

note_x = "Scales\nx: [-100,100]\ny: [0,10]\nz: [-10,15]"
note_z = "Scales\nx: [-10,10]\ny: [0,10]\nz: [-100,150]"

ax[0].scatter(x1[:,0], x1[:,1], c=labels_x)
ax[0].set_title("Increased x scale", fontsize=14)
ax[0].annotate(note_x, (0.15,0.4), xycoords = 'axes fraction', fontsize=13)
ax[1].scatter(z1[:,0], z1[:,1], c=labels_z)
ax[1].set_title("Increased z scale", fontsize=14)
ax[1].annotate(note_z, (0.15,0.4), xycoords = 'axes fraction', fontsize=13)

ax[0].get_xaxis().set_ticks([])
ax[0].get_yaxis().set_ticks([])
ax[1].get_xaxis().set_ticks([])
ax[1].get_yaxis().set_ticks([])

fig.show()












