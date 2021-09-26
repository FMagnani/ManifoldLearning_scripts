#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 17:35:23 2021

@author: fede
"""

# Utils
# Read and plot data 

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
    """
    Class to load training and label set, plot individual samples.
    
    Useful members:
        data: pandas DataFrame with label and vector components of the data point.
                
    Useful methods:    
        show_sample:
            pars: sample_number (int)
            returns: none
    """

    def __init__(self):    

        self.train_path = "data/mnist/train-images.idx3-ubyte"
        self.label_path = "data/mnist/train-labels.idx1-ubyte"
         
        # shape = (60000, 28, 28)
        self.data_matrix = idx2numpy.convert_from_file(self.train_path)
        
        n_samples, dim1, dim2 = self.data_matrix.shape
        vector_dim = dim1*dim2
        # shape = (60000, 784)
        self.data_vector = self.data_matrix.reshape(n_samples, vector_dim)
        
        self.data["labels"] = idx2numpy.convert_from_file(self.label_path)
        self.data = pd.DataFrame(self.data_vector)
        
    def show_sample(self, sample_number):
        
        plt.imshow(self.data_matrix[sample_number], cmap="gray")


# COIL20 dataset
class coil20():
    """
    Class to load coil20 dataset and show individual samples.
    
    Useful members:
        data: pandas DataFrame with object number, view number, vector components.

    Useful methods:
        save_data:
            pars: none, returns: none
            Save the 'data' member in csv format (fixed name).
        show_sample:
            pars: obj (int), view (int)
            returns: none 
    """
    
    def __init__(self):
        
        # Actually I don't know if it's faster than the other way.
        if (os.path.isfile("COIL20_vectorized.csv")):
            
            self.data = pd.read_csv("COIL20_vectorized.csv", index_col=0)
            
        else:
        
            self.coil20_path = "data/coil-20-proc/"
        
            row_list = []
            
            for im_path in glob.glob(self.coil20_path+"*.png")[:10]:
    
                # im_path is like 'data/coil-20-proc/obj9__41.png'
                obj = im_path.split("obj")[1].split("__")[0]
                view = im_path.split("obj")[1].split("__")[1].split('.')[0]
                
                obj = int(obj)
                view = int(view)
                
                im = np.array(imageio.imread(im_path)) # shape = (128,128)
                im = im.reshape(128*128) # shape = (128*128,)
            
                row = np.array([obj, view])
                row = np.concatenate((row, im))
                
                row_list.append(row)
                
            self.data = pd.DataFrame(row_list) 
            self.data.columns = np.concatenate( (["obj","view"], self.data.columns[:-2]) )
            
    def save_data(self):
        
        name = "COIL20_vectorized.csv"
        self.data.to_csv(name, index=True)

    def show_sample(self, obj, view):
        
        path = "data/coil-20-proc/obj"+str(obj)+"__"+str(view)+".png"
        
        im = img.imread(path)
        plt.imshow(im, cmap="gray")


# Swiss roll dataset
class swiss_roll():
    """
    Creates an istance of a swiss_roll dataset. 
    Parameters: 
        n_samples (int)
        noise (float) - Gaussian noise applied to each point
        seed (int) - Random seed to replicability
    
    Useful members:
        data: 3 dimensional points of the dataset. np.array with shape (n_samples, 3)
        t: The univariate position of the sample according to the main dimension of the points.

    Useful methods:
        plot
    """
    
    def __init__(self, n_samples, noise, seed):

        self.data, self.t = self.make_swiss_roll(n_samples, noise, seed)


    def plot(self):
        
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        
        x = self.data[:,0]
        y = self.data[:,1]
        z = self.data[:,2]
        
        ax.scatter(x,y,z, c = self.t, cmap = 'viridis')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        plt.show()
        
    
    def make_swiss_roll(self, n_samples, noise, seed):        

        generator = np.random.RandomState(seed)        

        t = 1.5 * np.pi * (1 + 2 * generator.rand(1, n_samples))
        x = t * np.cos(t)
        y = 21 * generator.rand(1, n_samples)
        z = t * np.sin(t)

        X = np.concatenate((x, y, z))
        X += noise * generator.randn(3, n_samples)
        X = X.T
        t = np.squeeze(t)

        return X, t















