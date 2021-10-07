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

from umap.umap_ import fuzzy_simplicial_set
from umap.spectral import spectral_layout

#%%
"""
The swiss roll dataset is created giving to one of the axis a much greater 
scale. Due to the symmetry of its shape, actually this makes a difference
if it's done to the y axis (along the width of the roll) on one hand, or to
either the x or z axis on the other hand. 
"""

dataset_x = swiss_roll(2000, 0, 123, 10,10,1)
dataset_z = swiss_roll(2000, 0, 123, 1,10,10)

dataset_x.plot()
dataset_z.plot()

x = dataset_x.data
labels_x = dataset_x.t

z = dataset_z.data
labels_z = dataset_z.t

#%%
"""
Embedding by Laplacian Eigenmap for 10 n_neighbors.
"""

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

#%%
"""
Embedding by UMAP for some values of n_neighbors.
"""

n_neighbors_list = [5, 10, 15]

fig, axes = plt.subplots(1,3)

for j in [0,1,2]:
    
    n_neighbors = n_neighbors_list[j]
    ax = axes[j]

    fuztop_graph, _, _, _ = fuzzy_simplicial_set(
                                                 x,
                                                 n_neighbors = n_neighbors,
                                                 random_state = 123456,
                                                 metric = 'euclidean',
                                                 metric_kwds={},
                                                 knn_indices=None,
                                                 knn_dists=None,
                                                 angular=False,
                                                 set_op_mix_ratio=1.0,
                                                 local_connectivity=1.0,
                                                 apply_set_operations=True,
                                                 verbose=False,
                                                 return_dists=False,
                                                 )

    fuztop_embedding = spectral_layout(x, fuztop_graph, dim=2,
                                       random_state=123456, 
                                       metric='euclidean', metric_kwds={})


    ax.scatter(fuztop_embedding[:,0], fuztop_embedding[:,1], c=labels_x)

    ax.set_title("N_neighbors: "+str(n_neighbors))
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


note = "Scales\nx: [-100,100]\ny: [0,10]\nz: [-10,15]"
axes[1].annotate(note_x, (0.1,0.8), xycoords = 'axes fraction', fontsize=13)

fig.show()











