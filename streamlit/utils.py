import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from keras import Model


def convert_picture(im):

    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    inverted = 255 - grey
    img = cv2.resize(inverted, (28, 28  ))
    img = np.reshape(img, (1, 28, 28))
    return img

def trim_img(img, plot=False):
    # print(img)
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
        # x_diff = int((x-y)/2)
        filler = np.zeros((x, x_diff))
        pic = np.hstack([filler, pic, filler])

    x,y = pic.shape
    y_diff = int((y-x)/2)+add
    if y_diff > 0:
        filler = np.zeros((y_diff, y))
        pic = np.vstack([filler, pic, filler])

    pic = cv2.resize(pic, dsize=(28,28), interpolation=cv2.INTER_AREA)
    
    # plt.subplot(1,2,1)
    # plt.imshow(pic, cmap='gray')
    # # plt.xticks([]), plt.yticks([])
    # plt.subplot(1,2,2)
    # plt.imshow(img, cmap='gray')
    # plt.xticks([]), plt.yticks([])
    return pic

def fix_image(image, add=2, plot=False):
    pic = trim_img(image)
    pic = square_pick(pic, add, plot)
    return pic

def get_pic(img, add=0):
    # pic =cv2.imread(paths)
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

class Model_Class():
    def __init__(self,
                 model_path=''):

        self.model = self.load_model(model_path)
        # self.model = self.__get_model()
        self.conv_layers, self.layer_names = self.__get_convlayers()
        self.preds = ""

    def load_model(self, path):
        self.model = keras.models.load_model(path)
        return self.model

    def grad_cam(self, image, layer=None):
        self.preds = self.predict(image)

        if type(layer) is list:
            heatmap_list, superimposed_list = {},{}
            for layer_num in layer:
                heatmap = self.make_gradcam_heatmap(np.expand_dims(image, axis=0), layer_num, np.argmax(self.preds[0]))
                superimposed_img = self.superimpose(image,heatmap)
                heatmap_list[self.model.layers[layer_num]._name] = heatmap
                superimposed_list[self.model.layers[layer_num]._name] = superimposed_img
            return image, heatmap_list, superimposed_list
        else:
            heatmap = self.make_gradcam_heatmap(np.expand_dims(image, axis=0), layer, np.argmax(self.preds[0]))
            superimposed_img = self.superimpose(image,heatmap)

            heatmap = cv2.resize(heatmap, dsize=(28,28))
            return image, heatmap, superimposed_img


    def make_gradcam_heatmap(self, img_array, layer=None, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        if layer == None: layer=self.conv_layers[-1]
        model = self.model
        grad_model = Model(
            [model.inputs], [model.layers[layer].output, model.output]
            # [model.inputs], [model.get_layer(self.layer_names[-1]).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def predict(self, pic):
        x = cv2.resize(pic, dsize=(28,28))
        # x = model_preprocess(pic)
        if len(x.shape) < 4:
          x = np.expand_dims(x, axis=0)

        preds = self.model.predict(x)
        return preds

    def superimpose(self, pic,heatmap):
        # img_numpy = np.asarray(np.clip(pic, 0, 190))
        pic[pic > 0.4] = 1.0
        img_numpy = np.asarray(pic)
        # plt.imshow(pic)
    
        heatmap_resized = cv2.resize(heatmap, (img_numpy.shape[1], img_numpy.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        superimposed_img = 0.3*heatmap_resized[:,:,::-1] + img_numpy
        
        superimposed_img = superimposed_img.astype(np.uint8)
        return superimposed_img

    def __get_convlayers(self):
        list_conv_layers = []
        list_layer_names = []
        for i,l in enumerate(self.model.layers):
            # print(str(l).split('.'))
            if str(l).split('.')[2] == 'convolutional':
                list_conv_layers.append(i)
                list_layer_names.append(l._name)
        return list_conv_layers, list_layer_names

def plot_gradcam(image, heatmap, superimposed_img):
  if type(heatmap) == dict and type(superimposed_img) == dict:
      nlen = len(heatmap)
      fig, ax = plt.subplots(nlen,3, figsize=(10, nlen*3.5))
      # fig.figsize=(20,20)
      for i, (k, img) in enumerate(heatmap.items()):
          ax[i,0].imshow(img+50)
          ax[i,1].set_title(k)
          ax[i,1].imshow(superimposed_img[k])
          ax[i,2].imshow(image)
  else:
    plt.subplot(1,3, 1)
    plt.imshow(heatmap)
    plt.subplot(1,3, 2)
    plt.imshow(superimposed_img)
    plt.subplot(1,3, 3)
    plt.imshow(image)