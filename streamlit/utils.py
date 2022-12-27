import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import cv2

from pathlib import Path

def get_root_path(folder_name:str) -> Path:
    """uses foldername to locate the project root path

    Args:
        folder_name (str): foldername to find in the path

    Returns:
        Path: _description_
    """    
    
    root_path = Path(os.getcwd())    
    for i in range(len(Path(os.getcwd()).parts)-1):
        
        if root_path.parents[i].name == folder_name:
            return root_path.parents[i], root_path.parents[i] / 'data'
            
def transp(image):
    """Reshapes and transposes image. Needed for emnist

    Args:
        image (_type_): image as numpy array.

    Returns:
        _type_: transposed an resized image
    """    
    image = image.reshape([28, 28])
    image = np.transpose(image)
    # image = np.rot90(image)
    return image

def gen_sets(p:Path, lower:int=0, upper:int=0, numlist:list=None) -> np.array:
    """Generates dataset filters on either range or list of labels

    Args:
        p (_type_): _description_
        lower (int, optional): Lower label to use. Defaults to 0.
        upper (int, optional): Upper label to use. Defaults to 0.
        numlist (_type_, optional): List of labels to use. Defaults to None.

    Returns:
        _type_: features and labels in separate numpy arrays
    """    
    
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

def unique_bar(dataset:np.array, key:dict):
    """Makes barplots of number or counts of unique.

    Args:
        dataset (_type_): Dataset of labels
        key (_type_): Dict to decode numbers to lables
    """    
    x, y = np.unique(dataset, return_counts=True)
    sns.barplot(x=[key[xi] for xi in x], y=y)
    
def convert_picture(im:np.array)-> np.array:

    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    inverted = 255 - grey
    img = cv2.resize(inverted, (28, 28  ))
    img = np.reshape(img, (1, 28, 28))
    return img

def trim_img(img) -> np.array:
    """Trims black borders from image

    Args:
        img (_type_): Picture to be trimmed

    Returns:
        np.array: Trimmed image
    """    
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

def square_pick(img:np.array, add:int=0) -> np.array:
    """Tries to make the image square

    Args:
        img (np.array): Image as numpy array
        add (int, optional): How much black border to add. Defaults to 0.

    Returns:
        np.array: Image with borders.
    """    
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


def fix_image(image:np.array, add:int=0) -> np.array:
    """Trims and squares image

    Args:
        image (np.array): Picture
        add (int, optional): Sets how much dark border to be used. Defaults to 0.

    Returns:
        np.array: Trimmed and squared image
    """    
    pic = trim_img(image)
    pic = square_pick(pic, add)
    return pic
