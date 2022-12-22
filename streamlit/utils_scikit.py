import cv2
import numpy as np

def prepping(im):
    emnist_size = 28
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(grey, (emnist_size, emnist_size))
    arr = img
    
    arr[arr < 125] = 0
    arr[arr != 0] = 1

    a = np.sum(arr, axis=0)
    b = np.sum(arr, axis=1)
    print(a,b)
    left = np.where(a != 0)[0][0]
    right = np.where(np.flip(a)!=0)[0][0]
    top = np.where(b != 0)[0][0]
    bottom = np.where(np.flip(b)!=0)[0][0]
    
    arr = arr[top:emnist_size - bottom, left: emnist_size - right]
    top_pad = int((28- arr.shape[0])/2)
    bottom_pad = top_pad
    if (top_pad * 2 + arr.shape[0]) < emnist_size: bottom_pad += 1 
    left_pad = int((emnist_size- arr.shape[1])/2)
    right_pad = left_pad
    if (left_pad * 2 + arr.shape[1]) < emnist_size: right_pad += 1 
    
    padded = np.pad(arr, pad_width=((top_pad, bottom_pad), (left_pad, right_pad)))
    print(padded)
    
    padded = padded.reshape(-1, emnist_size, emnist_size, 1)
    return padded