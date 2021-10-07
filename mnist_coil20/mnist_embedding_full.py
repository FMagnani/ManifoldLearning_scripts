#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:19:30 2021

@author: fede
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # For custom legend
from matplotlib.colors import Normalize # For custom legend

# Make data

fig, axes = plt.subplots(3, 3)

neighbors_list = [5, 80, 300]

for j in [0,1,2]:

  ax = axes[:,j]

  n_neighbors = neighbors_list[j]

  fuztop_graph, sigmas, rhos, dists = umap.umap_.fuzzy_simplicial_set(
                                                        data,
                                                        n_neighbors = 15,
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
                                                        return_dists=True,
                                                        )

  dists = sparse.find(dists)[2]
  weights = sparse.find(fuztop_graph)[2]

  annotation = "Distances mean:"+str(np.mean(dists))[:4]+", std:"+str(np.std(dists))[:4] +\
              "\nSigma mean:"+str(np.mean(sigmas))[:4]+", std:"+str(np.std(sigmas))[:4] +\
              "\nWeigths mean:"+str(np.mean(weights))[:4]+", std:"+str(np.std(weights))[:4]

  annotation = "Weigths\nmean:"+"123456"+"\nStd:"+"123456"
  
  x_label =    "Sigma:"+"123456"+"'\xB1'"+"123456"


  fuztop_embedding = spectral_layout(data, fuztop_graph, dim=2,
                                    random_state=123456, metric='euclidean', metric_kwds={})

  umap_embedding = umap.UMAP(n_neighbors=n_neighbors, n_components=2, metric='euclidean').fit_transform(data)


  ax[0].hist(weights, bins=60, color='black')
  ax[1].scatter(fuztop_embedding[:,0], fuztop_embedding[:,1],
                c = labels, cmap='tab10', marker='+', s=25)
  ax[2].scatter(umap_embedding[:,0], umap_embedding[:,1],
                c = labels, cmap='tab10', marker='+', s=25)

  for i in range(3):
    ax[i].get_xaxis().set_ticks([])
    ax[i].get_yaxis().set_ticks([])

  ax[0].set_title("N neighbors = "+str(n_neighbors))
  ax[0].annotate(annotation, (0.5,0.6), xycoords='axes fraction')
  ax[0].set_xlabel(x_label)


# Legend
cmap = plt.cm.tab10
norm = Normalize(vmin=0, vmax=9)
legend_list = []
for c in range(10):
  color = cmap(norm(c))
  patch = mpatches.Patch(color=color, label=str(c))
  legend_list.append(patch)

axes[2,1].legend(handles=legend_list, 
                 loc='upper center', bbox_to_anchor=(0.5, 1.24),
                 ncol=10, fancybox=True, shadow=True)

axes[0,0].set_ylabel("Weights\ndistribution")
axes[1,0].set_ylabel("Fuzzy topological\nembedding")
axes[2,0].set_ylabel("UMAP\nembedding")

plt.show()

