import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

def get_root_path(folder_name):
    
    root_path = Path(os.getcwd())    
    for i in range(len(Path(os.getcwd()).parts)-1):
        
        if root_path.parents[i].name == folder_name:
            return root_path.parents[i], root_path.parents[i] / 'data'
            
def transp(image):
    image = image.reshape([28, 28])
    image = np.transpose(image)
    # image = np.rot90(image)
    return image

def gen_sets(p, lower=0, upper=0, numlist=None):
    
    x = np.genfromtxt(p, delimiter=',')
    if not (lower == 0 and upper == 0 and numlist==None):
        if numlist == None:
            x = x[(x[:,0] >= lower) & (x[:,0] <= upper)]
        else:
            x = x[np.in1d(x[:,0] , numlist)]
    y = x[:,[0]].astype('int')

    y = y.reshape(x.shape[0])
    x = x[:,1:]
    
    x = np.apply_along_axis(transp, 1, x)
    return x, y

def unique_bar(dataset, key):
    x, y = np.unique(dataset, return_counts=True)
    sns.barplot(x=[key[xi] for xi in x], y=y)
    
