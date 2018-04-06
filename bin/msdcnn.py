# coding: utf-8

import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation,Reshape
from keras import backend as K
from keras.layers.merge import concatenate
from keras.layers import Lambda, Dot
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import argparse, sys, os, errno
from glob import glob
from keras.layers import add, concatenate
import h5py
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.initializers import RandomNormal
from keras.callbacks import ModelCheckpoint

# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
INPUT_CHANNELS = 16
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1


def binary_crossentropy_with_nan(y_true, y_pred):
    not_nan = tf.logical_not(tf.is_nan(y_true))
    y_true = tf.boolean_mask(y_true, not_nan)
    y_pred = tf.boolean_mask(y_pred, not_nan)
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
#keras.losses.binary_crossentropy_with_nan = binary_crossentropy_with_nan
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
#define new loss using output layer and hidden layer
class CrossEntropyLoss(object):
    def __init__(self, model,alpha):
        self.layer = model.get_layer(index = -8).output
        self.__name__ = 'CrossEntropyLoss'
        self.alpha = alpha
    def __call__(self,y_true, y_pred):
        return binary_crossentropy_with_nan(y_true, y_pred) + self.alpha * mean_squared_error(self.layer,tf.transpose(self.layer, perm=[0, 2, 1, 3]))
class MSE(object):
    def __init__(self, model):
        self.layer = model.get_layer(index = -8).output
        self.__name__ = 'MSE'
    def __call__(self,y_true,y_pred):
        return mean_squared_error(self.layer,tf.transpose(self.layer, perm=[0, 2, 1, 3]))
def binary_accuracy_with_nan(y_true, y_pred):
    not_nan = tf.logical_not(tf.is_nan(y_true))
    y_true = tf.boolean_mask(y_true, not_nan)
    y_pred = tf.boolean_mask(y_pred, not_nan)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
keras.metrics.binary_accuracy_with_nan = binary_accuracy_with_nan
def mse_with_nan(y_true, y_pred):
    not_nan = tf.logical_not(tf.is_nan(y_true))
    y_true = tf.boolean_mask(y_true, not_nan)
    y_pred = tf.boolean_mask(y_pred, not_nan)
    return K.mean(K.square(y_pred - y_true), axis=-1)
keras.metrics.mse_with_nan = mse_with_nan
keras.losses.mse_with_nan = mse_with_nan
alpha = 1.
#use matrix and its .T 's mse as part of the loss function
def loss(y_true, y_pred,alpha):
    mse =K.mean(K.square(y_pred - K.transpose(y_pred)), axis=-1)
    return binary_crossentropy_with_nan(y_true, y_pred) +alpha * mse


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)
def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection ) / (np.sum(y_true_f) + np.sum(y_pred_f))
def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
def mean_squared_error_np(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true), axis=-1)
def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)
def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 128. * K.mean(diff, axis=-1)
def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True
    return k
conv_init = RandomNormal(0, 0.02)
def mixed_scaled_dense_network(input_tensor, num_layers=16, nc_in=3, nc_out=1): #, nc_in=3, nc_out=1
    """
        Inefficient implementation of paper: A mixed-scale dense convolutional neural network for image analysis
        http://www.pnas.org/content/115/2/254
        """
    msd_layers = {}
    x = input_tensor
    msd_layers["input"] = x
    for i in range(num_layers):
        dilation = (i % 10) + 1
        msd_layers["layer{0}".format(i)] = Conv2D(1, kernel_size=3, strides=1, dilation_rate=dilation,
                                                  kernel_initializer=conv_init, padding="same")(x)
        for j in range(i):
            dilation = ((i + j) %10) + 1
            conv_3x3 = Conv2D(1, kernel_size=3, strides=1, dilation_rate=dilation,
                            kernel_initializer=conv_init, use_bias=True, padding="same")(msd_layers["layer{0}".format(j)])
            msd_layers["layer{0}".format(i)] = add([msd_layers["layer{0}".format(i)], conv_3x3])
        msd_layers["layer{0}".format(i)] = Activation("relu")(msd_layers["layer{0}".format(i)])
    concat_all = x
    for i in range(num_layers):
        concat_all = concatenate([concat_all, msd_layers["layer{0}".format(i)]])
    msd_layers["merge_concat_all"] = concat_all
    out = Conv2D(nc_out, kernel_size=1, kernel_initializer=conv_init, padding="same")(concat_all)
    msd_layers["output"] = out
    return out, msd_layers

def msdcnnmet(num_layers=16, nc_in=3, nc_out=1):
    if K.image_dim_ordering() == 'th':
        inputs = Input((INPUT_CHANNELS, 128,128))
        axis = 1
    else:
        inputs = Input((128, 128, INPUT_CHANNELS))
        axis = 3

#check msdcnn的最后一层是conv吗？？看起来是的
    conv_final , _ = mixed_scaled_dense_network(inputs,num_layers, nc_in, nc_out)

    final_row = Lambda(lambda x: K.sum(x, axis=1), output_shape=(128,))(conv_final)
    #sum by columns output a vector of 128 elements
    final_row_ = Reshape((128,), input_shape=(128,1))(final_row)
    final_row =Lambda(lambda x: x/128.0, output_shape=(128,))(final_row_)

    final_col = Lambda(lambda x: K.sum(x, axis=1), output_shape=(128,))(conv_final)
    #sum by columns output a vector of 128 elements
    final_col_ = Reshape((128,), input_shape=(128,2))(final_col)
    final_col =Lambda(lambda x: x/128.0, output_shape=(128,))(final_col_)

    final = concatenate([final_row,final_col],axis =-1)
    model = Model(inputs, final, name="ZF_UNET_128")
    return model
