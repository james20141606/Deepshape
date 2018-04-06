# coding: utf-8
#! /usr/bin/env python

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import argparse, sys, os, errno
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from tqdm import tqdm
import keras as K
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from unet_128_model_row_column import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
set_session(tf.Session(config=config))


model = UNET_128()
optim = Adam()
#model.summary()
#model.layers[-8].output
#model.get_layer(index = -8).output
model.compile(optimizer=optim, loss=CrossEntropyLoss(model,0), metrics=[binary_accuracy_with_nan,binary_crossentropy_with_nan,MSE(model)])
#model.summary()

number = 10000

with h5py.File('/home/chenxupeng/projects/deepshape/data/new/train_0') as f:
    images_train = f['train_images'][:number]
with h5py.File('/home/chenxupeng/projects/deepshape/data/new/Spitale_2015_invivo_CDS_0.1') as f:
    y_train = f['y_train'][:number]
y_train = np.concatenate([y_train,y_train],axis = 1)
#y size 256

#callbacks = [model_checkpoint]
model_checkpoint = ModelCheckpoint('newunet__row_col_weights_mse.hdf5', monitor='binary_accuracy_with_nan', save_best_only=True)
#EarlyStopping(monitor=binary_accuracy_with_nan, patience=10, verbose=0)
def Model(images_train,y_train,cv,seq_counts):
    model.fit(images_train, y_train, batch_size=32, nb_epoch=250,
              verbose=1, shuffle=True,validation_split=0.2,
              callbacks=[model_checkpoint,EarlyStopping(monitor='CrossEntropyLoss', patience=10, verbose=0),TensorBoard(log_dir='/home/chenxupeng/projects/deepshape/output/tensorboard/unet/log_dir')])
    model.save('output/newunet_row_col_mse_1.30_3.hdf5')

for i in range(1):
    Model(images_train[:number,:,:,:],y_train[:number,:],i,number)
