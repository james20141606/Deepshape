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
    model.add(GlobalMaxPooling1D())
    model.add(Flatten())
    model.add(Dense(64, kernel_regularizer=l2(0.0001)))
    model.add(Activation('relu'))
    output_size = window_size if dense else 1
    model.add(Dense(output_size, kernel_regularizer=l2(0.0001)))
    if not regression:
        model.add(Activation('sigmoid'))
    return model