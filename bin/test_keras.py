#! /usr/bin/env python
from __future__ import print_function
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from numba import jit
import keras
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.pooling import MaxPooling1D, MaxPooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D
from keras.regularizers import l2, l1, l1_l2
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.utils.vis_utils import model_to_dot

@jit
def make_convolutional(n_samples=10000, shape=(100, 4)):
    L = 10
    pattern = np.random.uniform(-1, 1, size=(L, shape[1]))
    X = np.random.normal(size=(n_samples, shape[0], shape[1]))
    y = np.random.randint(2, size=n_samples)
    for i in range(n_samples):
        if y[i] > 0:
            j = np.random.randint(shape[0] - L)
            X[i, j:(j + L)] = pattern
    return X, y

def build_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
    
def build_convolutional(input_shape):
    model = Sequential()
    model.add(Conv1D(16, 3, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def test_model(n_samples=10000, model='conv'):
    print('generate dataset')
    if model == 'mlp':
        X, y = make_classification(n_samples=n_samples, n_classes=2, n_features=128)
        model = build_model(input_shape=(128,))
    elif model == 'conv':
        X, y = make_convolutional(n_samples, shape=(128, 4))
        model = build_convolutional((128, 4))
    print('compile the model')
    model.compile(optimizer='Adam', 
        metrics=['binary_accuracy'],
        loss=['binary_crossentropy'])
    model.summary()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print('train the model')
    model.fit(X_train, y_train, batch_size=50, epochs=20)
    print('test the model')
    y_pred = model.predict(X_test, batch_size=50)
    y_pred_labels = (y_pred > 0.5).astype(np.int32)
    print('roc_auc_score = {:f}'.format(roc_auc_score(y_test, y_pred)))
    print('accuracy_score = {:f}'.format(accuracy_score(y_test, y_pred_labels)))
    
if __name__ == '__main__':
    test_model()
