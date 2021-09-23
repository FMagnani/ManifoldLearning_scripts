#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 17:35:23 2021

@author: fede
"""

# Utils
# Read and plot raw data 

import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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



        
        






















