import cv2
import numpy as np
import keras
from PIL import Image
from io import BytesIO


def prepping(im:np.array)-> np.array:
    """Preprocesses images for the classic ml models.
    By resizing cropping dark areas and binarize the picture.

    Args:
        im (np.array): img input in color or grayscale

    Returns:
        np.array: 28,28 picture in the form of np.array
    """    
    emnist_size = 28
    if len(im.shape) == 3:
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        grey = im
    img = cv2.resize(grey, (emnist_size, emnist_size))
    arr = img
    
    arr[arr < 125] = 0
    arr[arr != 0] = 1

    a = np.sum(arr, axis=0)
    b = np.sum(arr, axis=1)
    # print(a,b)
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
    # print(padded)
    
    padded = padded.reshape(-1, emnist_size, emnist_size, 1)
    return padded

def trim_img(img: np.array) -> np.array:
    """Trims image of dark borders

    Args:
        img (np.array): grayscale image as np array

    Returns:
        np.array: cropped image
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
    """Makes the picture square formed. 

    Args:
        img (np.array): grayscale picture as np.array
        add (int, optional): adds a border around the picture by set amount of pixels. Defaults to 0.

    Returns:
        np.array: picture as np.array
    """    
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

def fix_image(image:np.array, add:int=2) -> np.array:
    
    pic = trim_img(image)
    pic = square_pick(pic, add)
    return pic

def get_pic(img: np.array, add:int=0) -> np.array:
    pic = img
    if len(pic.shape) == 3:
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.resize(pic, (28, 28))
    
    pic = 255 - pic
    pic[pic < 150] = 0

    pic = fix_image(pic, add)
    pic = np.reshape(pic, (1, 28, 28))
    pic = pic.astype("float32") / 255
    pic = np.expand_dims(pic, -1)
    return pic

def get_nn_result(model:keras.Model, image:np.array, mm:dict, get_pic:function)->np.array:
    print(type(image))
    if type(image) != np.ndarray:
        img = get_pic(np.asarray(Image.open(BytesIO(image.getbuffer()))), 2)
    else:
        img = get_pic(image, 2)
    return mm[np.argmax(model.predict(img))]
    
def preprocess_camera_picture(img, non_zero=1, threshold=0.5):
    im = cv2.bilateralFilter(img,30,75,75)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    im[im < np.median(im)*threshold] = 0
    im[im > np.median(im)*threshold + 0.03] = 255

    im = cv2.GaussianBlur(im, (5,5),0)
    im = 255 -im
    # Crops the picture
    im = im[:, np.apply_along_axis(np.count_nonzero, 0, im) >= non_zero]
    im = im[np.apply_along_axis(np.count_nonzero, 1, im) >= non_zero, :]
    # im = cv2.resize(im, (100, 100))
    return im