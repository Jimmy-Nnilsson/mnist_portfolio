import cv2
from tensorflow import keras
import numpy as np

def convert_picture(im):

    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    inverted = 255 - grey
    img = cv2.resize(inverted, (28, 28  ))
    img = np.reshape(img, (1, 28, 28))
    # _,img = cv2.threshold(img, 40,255, cv2.THRESH_BINARY)
    # img = img.astype('bool')
    return img

