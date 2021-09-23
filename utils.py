#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 17:35:23 2021

@author: fede
"""

# Utils
# Read and plot raw data 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import idx2numpy # MNIST
import imageio #COIL20
import glob #COIL20
import matplotlib.image as img

# MNIST dataset

class mnist_data():

    def __init__(self):    

        self.mnist_path = "data/train-images.idx3-ubyte"
        self.data_matrix = idx2numpy.convert_from_file(self.mnist_path)
        
        n_samples, dim1, dim2 = self.data_matrix.shape
        vector_dim = dim1*dim2
        self.data_vector = self.data_matrix.reshape(n_samples, vector_dim)

    def show_sample(self, sample_number):
        
        plt.imshow(self.data_matrix[sample_number], cmap="gray")

# COIL20 dataset

class coil20_data():
    
    def __init__(self):
        
#        if (os.path.isfile("COIL20_vectorized.csv")):
        if(False):
            
            pass
          
            # self.coil20_df = pd.read_csv("COIL20_vectorized.csv", index_col=0,
            #                              dtype={"path":str, "vector":list})
            # self.data_vector = np.array(self.coil20_df["vector"].values.tolist())
            
        else:
        
            self.coil20_path = "data/coil-20-proc/"
        
            self.data = pd.DataFrame(columns=["object","view","path","vector"])

            obj_list = []
            view_list = []
            path_list = []
            vec_list = []
            for im_path in glob.glob(self.coil20_path+"*.png")[:10]:
                
                # im_path is like 'data/coil-20-proc/obj9__41.png'
                
                obj = im_path.split("obj")[1].split("__")[0]
                view = im_path.split("obj")[1].split("__")[1].split('.')[0]
                
                im = np.array(imageio.imread(im_path)) # shape = (128,128)
                im = im.reshape(128*128) # shape = (128*128,)
                
                obj_list.append(int(obj))
                view_list.append(int(view))
                path_list.append(im_path)
                vec_list.append(im)
                
            self.data["object"] = obj_list
            self.data["view"] = view_list
            self.data["path"] = path_list
            self.data["vector"] = vec_list

    def show_sample(self, obj, view):
        
        query1 = self.data["object"]==obj
        index_list = self.data.index[query1].tolist()
        query2 = self.data.loc[index_list]["view"]==view 
        index = self.data.index[query2].tolist()[0]
        
        path = self.data.loc[index]["path"]
        
        im = img.imread(path)
        plt.imshow(im, cmap="gray")

















