from __future__ import print_function
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.layers.merge import Concatenate, Add
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.pooling import MaxPooling1D, MaxPooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, AveragePooling1D
from keras.regularizers import l2, l1, l1_l2
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras import backend as K

def basic(window_size, regression=False, dense=False):
    model = Sequential()
    model.add(Flatten(input_shape=(window_size, 4)))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    output_size = window_size if dense else 1
    model.add(Dense(output_size))
    if not regression:
        model.add(Activation('sigmoid'))
    return model

def conv1(window_size, regression=False, dense=False):
    model = Sequential()
    model.add(Conv1D(64, 5, padding='valid', input_shape=(window_size, 4), kernel_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 4, padding='valid', kernel_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 4, padding='valid', kernel_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(64, kernel_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    output_size = window_size if dense else 1
    model.add(Dense(output_size, kernel_regularizer=l2(0.0001)))
    if not regression:
        model.add(Activation('sigmoid'))
    return model

def mlp1(window_size, regression=False, dense=False):
    model = Sequential()
    model.add(Flatten(input_shape=(window_size, 4)))
    model.add(Dense(64, kernel_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    model.add(Dense(64, kernel_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    output_size = window_size if dense else 1
    model.add(Dense(output_size, kernel_regularizer=l2(0.0001)))
    if not regression:
        model.add(Activation('sigmoid'))
    return model

def resnet1(window_size, regression=False, dense=False):
    input = Input(shape=(window_size, 4))
    x = Conv1D(64, 5, padding='same', activation='relu')(input)
    for i in range(5):
        output = Conv1D(64, 3, padding='same', activation='relu')(x)
        output = Conv1D(64, 3, padding='same')(output)
        output = Add()([x, output])
        output = Activation('relu')(output)
        x = output
    output = AveragePooling1D(2)(output)
    output = Flatten()(output)
    output_size = window_size if dense else 1
    output = Dense(output_size)(output)
    if not regression:
        output = Activation('sigmoid')(output)
    model = Model(inputs=[input], outputs=[output])
    return model

def blstm3(window_size, regression=False, dense=False):
    model = Sequential()
    model.add(Bidirectional(LSTM(150, return_sequences=True), input_shape=(window_size, 4), merge_mode='ave'))
    model.add(Bidirectional(LSTM(1, return_sequences=True), merge_mode='ave'))
    model.add(Flatten())
    output_size = window_size if dense else 1
    model.add(Dense(output_size, kernel_regularizer=l2(0.0001)))
    if regression:
        model.add(Activation('sigmoid'))
    return model
