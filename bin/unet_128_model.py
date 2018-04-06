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


# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
INPUT_CHANNELS = 16
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1



def binary_crossentropy_with_nan(y_true, y_pred):
    not_nan = tf.logical_not(tf.is_nan(y_true))
    y_true = tf.boolean_mask(y_true, not_nan)
    y_pred = tf.boolean_mask(y_pred, not_nan)
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
keras.losses.binary_crossentropy_with_nan = binary_crossentropy_with_nan

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


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

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

def double_conv_layer(x, size, dropout, batch_norm):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    return conv


dropout_val = 0.2
def UNET_128(dropout_val=dropout_val, batch_norm=True):
    if K.image_dim_ordering() == 'th':
        inputs = Input((INPUT_CHANNELS, 128,128))
        axis = 1
    else:
        inputs = Input((128, 128, INPUT_CHANNELS))
        axis = 3
    filters = 32

    conv_128 = double_conv_layer(inputs, filters, dropout_val, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2, 2))(conv_128)

    conv_64 = double_conv_layer(pool_64, 2*filters, dropout_val, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2, 2))(conv_64)

    conv_32 = double_conv_layer(pool_32, 4*filters, dropout_val, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2, 2))(conv_32)

    conv_16 = double_conv_layer(pool_16, 8*filters, dropout_val, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2, 2))(conv_16)

    conv_8 = double_conv_layer(pool_8, 16*filters, dropout_val, batch_norm)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_8)

    conv_4 = double_conv_layer(pool_4, 32*filters, dropout_val, batch_norm)

    up_8 = concatenate([UpSampling2D(size=(2, 2))(conv_4), conv_8], axis=axis)
    up_conv_8 = double_conv_layer(up_8, 16*filters, dropout_val, batch_norm)

    up_16 = concatenate([UpSampling2D(size=(2, 2))(up_conv_8), conv_16], axis=axis)
    up_conv_16 = double_conv_layer(up_16, 8*filters, dropout_val, batch_norm)

    up_32 = concatenate([UpSampling2D(size=(2, 2))(up_conv_16), conv_32], axis=axis)
    up_conv_32 = double_conv_layer(up_32, 4*filters, dropout_val, batch_norm)

    up_64 = concatenate([UpSampling2D(size=(2, 2))(up_conv_32), conv_64], axis=axis)
    up_conv_64 = double_conv_layer(up_64, 2*filters, dropout_val, batch_norm)

    up_128 = concatenate([UpSampling2D(size=(2, 2))(up_conv_64), conv_128], axis=axis)
    up_conv_128 = double_conv_layer(up_128, filters, 0, batch_norm)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_128)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)

    final = Lambda(lambda x: K.sum(x, axis=1), output_shape=(128,))(conv_final)
    #sum by columns output a vector of 128 elements
    final_ = Reshape((128,), input_shape=(128,1))(final)
    final =Lambda(lambda x: x/128.0, output_shape=(128,))(final_)

    model = Model(inputs, final, name="ZF_UNET_128")
    return model
