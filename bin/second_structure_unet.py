# coding: utf-8


import numpy as np
import keras
from keras.models import Model
from keras import losses
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


def weighthed_binary_ce_loss_nan(alpha):
    #not_nan = tf.logical_not(tf.is_nan(y_true))
    def celoss(y_true, y_pred):
        zero = K.equal(y_true, K.zeros((1,)))
        one = K.equal(y_true, K.ones((1,)))
        weights = np.array([1, alpha])
        y_true_0 = tf.boolean_mask(y_true, zero)
        y_pred_0 = tf.boolean_mask(y_pred, zero)
        y_true_1 = tf.boolean_mask(y_true, one)
        y_pred_1 = tf.boolean_mask(y_pred, one)
        loss_0 = - y_true_0 * K.log(y_pred_0) * K.variable(weights[0])
        loss_1 = - y_true_1 * K.log(y_pred_1) * K.variable(weights[1])
        loss = K.mean(loss_0, -1) + K.mean(loss_1, -1)
        return loss
    return celoss
keras.losses.weighthed_binary_ce_loss_nan = weighthed_binary_ce_loss_nan


def weighthed_mae_loss_nan(alpha):
    #not_nan = tf.logical_not(tf.is_nan(y_true))
    def maeloss(y_true, y_pred):
        zero = K.equal(y_true, K.zeros((1,)))
        one = K.equal(y_true, K.ones((1,)))
        weights = np.array([1, alpha])
        y_true_0 = tf.boolean_mask(y_true, zero)
        y_pred_0 = tf.boolean_mask(y_pred, zero)
        y_true_1 = tf.boolean_mask(y_true, one)
        y_pred_1 = tf.boolean_mask(y_pred, one)
        loss_0 = K.mean(K.abs(y_pred_0 - y_true_0), axis=-1) * \
            K.variable(weights[0])
        loss_1 = K.mean(K.abs(y_pred_1 - y_true_1), axis=-1) * \
            K.variable(weights[1])
        return loss_0+loss_1
    return maeloss


keras.losses.weighthed_mae_loss_nan = weighthed_mae_loss_nan


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def binary_crossentropy_squeeze(y_true, y_pred):
    loss = K.mean(K.binary_crossentropy(
        K.flatten(y_true), K.flatten(y_pred)), axis=-1)
    return loss

class MSE(object):
    def __init__(self, model):
        self.layer = model.get_layer(index = -8).output
        self.__name__ = 'MSE'
    def __call__(self,y_true,y_pred):
        return mean_squared_error(self.layer,tf.transpose(self.layer, perm=[0, 2, 1, 3]))


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


dropout_val = 0.1


def UNET_128(dropout_val=dropout_val, batch_norm=True):
    inputs = Input((None, None, INPUT_CHANNELS))
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

    up_16 = concatenate(
        [UpSampling2D(size=(2, 2))(up_conv_8), conv_16], axis=axis)
    up_conv_16 = double_conv_layer(up_16, 8*filters, dropout_val, batch_norm)

    up_32 = concatenate(
        [UpSampling2D(size=(2, 2))(up_conv_16), conv_32], axis=axis)
    up_conv_32 = double_conv_layer(up_32, 4*filters, dropout_val, batch_norm)

    up_64 = concatenate(
        [UpSampling2D(size=(2, 2))(up_conv_32), conv_64], axis=axis)
    up_conv_64 = double_conv_layer(up_64, 2*filters, dropout_val, batch_norm)

    up_128 = concatenate(
        [UpSampling2D(size=(2, 2))(up_conv_64), conv_128], axis=axis)
    up_conv_128 = double_conv_layer(up_128, filters, 0, batch_norm)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_128)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('sigmoid', name='last_activation')(conv_final)

    #to use class weight the dimension must be no more than 3d
    flatten = Lambda(lambda x: K.batch_flatten(x))(conv_final)
    #soft = Lambda(lambda x: Dense(x,2, activation='softmax'))(flatten)
    #flatten =Flatten(input_shape = (None,None,1))(conv_final)
    model = Model(inputs, flatten, name="ZF_UNET_128")
    
    return model
