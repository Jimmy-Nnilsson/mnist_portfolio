import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from PIL import Image
from io import BytesIO


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

def square_pick(img, add=0, plot=False):
    pic = img.copy()
    x,y = pic.shape

    x_diff = int((x-y)/2)+add
    if x_diff > 0:
        filler = np.zeros((x, x_diff))
        pic = np.hstack([filler, pic, filler])

    x,y = pic.shape
    y_diff = int((y-x)/2)+add
    if y_diff > 0:
        filler = np.zeros((y_diff, y))
        pic = np.vstack([filler, pic, filler])

    pic = cv2.resize(pic, dsize=(28,28), interpolation=cv2.INTER_AREA)
    return pic

def fix_image(image, add=2, plot=False):
    pic = trim_img(image)
    pic = square_pick(pic, add, plot)
    return pic

def get_pic(img, add=0):
    pic = img
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.resize(pic, (28, 28))
    
    pic = 255 - pic
    pic[pic < 150] = 0

    pic = fix_image(pic, add)
    pic = np.reshape(pic, (1, 28, 28))
    pic = pic.astype("float32") / 255
    pic = np.expand_dims(pic, -1)
    return pic

def get_nn_result(model, image, mm, get_pic):
    print(type(image))
    if type(image) != np.ndarray:
        img = get_pic(np.asarray(Image.open(BytesIO(image.getbuffer()))), 2)
    else:
        img = get_pic(image, 2)
    return mm[np.argmax(model.predict(img))]

class EMNISTDataPreparation(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        all_columns = []
        self.image_size = 28
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        if type(X) != np.ndarray:
        
            np_X = np.asarray(Image.open(BytesIO(X.getbuffer())))       
            np_X = cv2.cvtColor(np_X, cv2.COLOR_BGR2GRAY) 
        else:
            np_X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY) 

        ret_lst = []
        
        np_X = np.array(cv2.resize(np_X, (self.image_size, self.image_size)))
        np_X = ~np_X
        np_X = np.array(np_X).reshape(-1, self.image_size, self.image_size)
        
        for idx in range(np_X.shape[0]):
            arr = np_X[idx, :, :]
            # arr = np.transpose(arr)
            arr[arr < 125] = 0
            arr[arr != 0]  = 1

            a = np.sum(arr, axis=0)
            b = np.sum(arr, axis=1)
            left = np.where(a != 0)[0][0]
            right = np.where(np.flip(a) != 0)[0][0]
            top = np.where(b != 0)[0][0]
            bottom = np.where(np.flip(b) != 0)[0][0]
            
            arr = arr[top:self.image_size - bottom, left:self.image_size - right]
            top_pad = int((self.image_size - arr.shape[0]) / 2)
            bottom_pad = top_pad
            if (top_pad * 2 + arr.shape[0]) < self.image_size: 
                bottom_pad +=1 
            left_pad = int((self.image_size - arr.shape[1]) / 2)
            right_pad = left_pad
            if (left_pad * 2 + arr.shape[1]) < self.image_size: 
                right_pad +=1 
            
            padded = np.pad(arr, pad_width=((top_pad, bottom_pad), (left_pad, right_pad)))
            ret_lst.append(padded)
        return_array = np.array(ret_lst).reshape(self.image_size,self.image_size,-1)# ** 2)
        return return_array