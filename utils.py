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
class mnist():

    def __init__(self):    

        self.mnist_path = "data/train-images.idx3-ubyte"
        self.data_matrix = idx2numpy.convert_from_file(self.mnist_path)
        
        n_samples, dim1, dim2 = self.data_matrix.shape
        vector_dim = dim1*dim2
        self.data_vector = self.data_matrix.reshape(n_samples, vector_dim)

    def show_sample(self, sample_number):
        
        plt.imshow(self.data_matrix[sample_number], cmap="gray")

# COIL20 dataset
class coil20():
    
    def __init__(self):
        
        if (os.path.isfile("COIL20_vectorized.csv")):
            
            self.data = pd.read_csv("COIL20_vectorized.csv", index_col=0)
            
        else:
        
            self.coil20_path = "data/coil-20-proc/"
        
            dict_list = []
            
            for im_path in glob.glob(self.coil20_path+"*.png"):
                
                # im_path is like 'data/coil-20-proc/obj9__41.png'
                
                obj = im_path.split("obj")[1].split("__")[0]
                view = im_path.split("obj")[1].split("__")[1].split('.')[0]
                
                im = np.array(imageio.imread(im_path)) # shape = (128,128)
                im = im.reshape(128*128) # shape = (128*128,)
                
                row = {"obj":int(obj), "view":int(view)}
                for (value, index) in zip(im, range(len(im))):
                    row[str(index)] = value
                
                dict_list.append(row)
                
            self.data = pd.DataFrame(dict_list)
            
    def save_data(self):
        
        name = "COIL20_vectorized.csv"
        self.data.to_csv(name, index=True)

    def show_sample(self, obj, view):
        
        path = "data/coil-20-proc/obj"+str(obj)+"__"+str(view)+".png"
        
        im = img.imread(path)
        plt.imshow(im, cmap="gray")

















