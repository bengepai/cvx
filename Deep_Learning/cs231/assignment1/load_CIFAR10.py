from __future__ import print_function
from k_nearest_neighbor import KNearestNeighbor
import random
import os
import numpy as np
import matplotlib.pyplot as plt

import pickle

def load_file(file,cifar10_dir):
    file = cifar10_dir + '\\' + file
    with open(file,'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

def load_CIFAR10(cifar10_dir):
    X_train = np.zeros((50000,3072))
    Y_train = np.zeros((50000))
    X_test = np.zeros((10000,3072))
    Y_test = np.zeros((10000))
    for root,dirs,files in os.walk(cifar10_dir):
        index = 0
        for file in files:
            if file != 'test_batch':
                data = load_file(file,cifar10_dir)
                X_train[index*10000:(index+1)*10000] = data[b'data']
                Y_train[index*10000:(index+1)*10000] = data[b'labels']
                index += 1
            else :
                X_test[0:10000] = data[b'data']
                Y_test[0:10000] = data[b'labels']
    return X_train,Y_train,X_test,Y_test

