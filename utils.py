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
          
            # self.coil20_df = pd.read_csv("COIL20_vectorized.csv", index_col=0,
            #                              dtype={"path":str, "vector":list})
            # self.data_vector = np.array(self.coil20_df["vector"].values.tolist())
            
        else:
        
            self.coil20_path = "data/coil-20-proc/"
        
            self.data_dict = {}
            for im_path in glob.glob(self.coil20_path+"*.png")[:10]:
            
                im = np.array(imageio.imread(im_path))
                im_list.append(im)
                
                self.data_dict.update({im_path:im})
                
            self.data_matrix = np.array(im_list)            
            
            n_samples, dim1, dim2 = self.data_matrix.shape
            vector_dim = dim1*dim2
            self.data_vector = self.data_matrix.reshape(n_samples, vector_dim)
  
            self.coil20_df["path"] = path_list
            self.coil20_df["vector"] = self.data_vector.tolist()

            self.coil20_df.to_csv("COIL20_vectorized.csv")


    def show_sample(self, sample_number):
        
        
        
        plt.imshow(self.data_matrix[sample_number], cmap="gray")


















