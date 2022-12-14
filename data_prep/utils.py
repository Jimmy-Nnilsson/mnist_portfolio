import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import cv2

from pathlib import Path

def get_root_path(folder_name):
    root_path = Path(os.getcwd())    
    for i in range(len(Path(os.getcwd()).parts)-1):
        if root_path.parents[i].name == folder_name:
            return root_path.parents[i], root_path.parents[i] / 'data'
            
def transp(image):
    image = image.reshape([28, 28])
    image = np.transpose(image)
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
    
def convert_picture(im):
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    inverted = 255 - grey
    img = cv2.resize(inverted, (28, 28  ))
    img = np.reshape(img, (1, 28, 28))
    return img

def trim_img(img):
    x,y = img.shape
    pic = img
    rem_x, rem_y = [],[]
    
    for i in reversed(range(max(x,y))):
        
        if not np.sum(pic[i,0:]) == 0:
            rem_y.append(i)
        if not np.sum(pic[0:,i]) == 0:
            rem_x.append(i)

    pic = pic[min(rem_y):max(rem_y),:]
    pic = pic[:,min(rem_x):max(rem_x)]
    return pic

def square_pick(img, add=0):
    pic = img.copy()
    x,y = pic.shape

    x_diff = int((x-y)/2)+add
    if x_diff > 0:
        # x_diff = int((x-y)/2)
        filler = np.zeros((x, x_diff))
        pic = np.hstack([filler, pic, filler])

    x,y = pic.shape
    y_diff = int((y-x)/2)+add
    if y_diff > 0:
        filler = np.zeros((y_diff, y))
        pic = np.vstack([filler, pic, filler])

    pic = cv2.resize(pic, dsize=(28,28), interpolation=cv2.INTER_AREA)
    

    return pic


def fix_image(image, add=0):
    pic = trim_img(image)
    pic = square_pick(pic, add)
    return pic
