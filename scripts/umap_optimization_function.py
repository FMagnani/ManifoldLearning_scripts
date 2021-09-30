#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 16:46:03 2021

@author: fede
"""

import numpy as np
#import matplotlib.pyplot as plt
import umap
import umap.plot
from utils import swiss_roll

#%%

"""
SCRIPT useful in the case that it's needed to optimize a layout through the
internal function of umap. 
Not very handy since the computation of the various parameters is done via the
umap class, in any case. 
"""

# Create data

seed = 123456

n_samples = 2000
noise = 0

xsc =1
ysc =100
zsc =1

data = swiss_roll(n_samples, noise, seed, xsc,ysc,zsc)

X = data.data
color = data.t

#%%

n_neighbors = 15

# Fit model in order to retrieve the parameters for the optimization
model = umap.UMAP(n_neighbors, n_components=2, metric='euclidean')
model = model.fit(X)            

#%%

# Manually optimize calling the internal function

# Weighted graph
fuztop_graph,_,_ = umap.umap_.fuzzy_simplicial_set(X, n_neighbors=n_neighbors,
                                               metric='euclidean',
                                               random_state=seed, verbose=True)
# Low dimensional optimization
# Equal to model.embedding_ up to rotation and scale
if model.n_epochs is None:
    n_epochs = -1
else:
    n_epochs = np.max(model.n_epochs)

opt_emb, _ = umap.umap_.simplicial_set_embedding(X, fuztop_graph, n_components=2,
                                                 initial_alpha = model._initial_alpha,
                                                 a = model._a,
                                                 b = model._b,
                                                 gamma = model.repulsion_strength,
                                                 negative_sample_rate=model.negative_sample_rate,
                                                 n_epochs=n_epochs,
                                                 init='spectral',
                                                 random_state=umap.umap_.check_random_state(seed),
                                                 metric="euclidean",
                                                 metric_kwds={},
                                                 densmap=model.densmap,
                                                 densmap_kwds=model._densmap_kwds,
                                                 output_dens=model.output_dens,
                                                 )






