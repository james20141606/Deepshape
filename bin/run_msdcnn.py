# coding: utf-8
#! /usr/bin/env python

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import argparse, sys, os, errno
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Conv2D
from keras.models import Model
from keras.layers import Input, Activation
from keras.layers import add, concatenate
import seaborn as sns
import h5py
import keras as K
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.initializers import RandomNormal
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from msdcnn import *
from keras.optimizers import Adam,Adadelta,Adagrad,RMSprop,SGD,Adamax,Nadam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser()
parser.add_argument('-r', dest='rate')
parser.add_argument('-i', dest='ncin')
parser.add_argument('-o', dest='ncout')
parser.add_argument('-n', dest='layernums')
parser.add_argument('-s', dest='datasize')
parser.add_argument('-m', dest='optname')
args = parser.parse_args()
'''
python run_msdcnn.py \
    -r 1 \
    -i 3 \
    -o 1 \
    -n 20 \
    -s 400 \
    -m adadelta
'''
model = msdcnnmet(num_layers=int(args.layernums), nc_in=int(args.ncin), nc_out=int(args.ncout))   #in 3 out 1

def optimize(rate,name):
    lr =float(rate)
    if name =='adam':
        optim = Adam(lr)
    if name =='adadelta':
        optim = Adadelta(lr)
    if name =='adagrad':
        optim = Adagrad(lr)
    if name =='rmsprop':
        optim = RMSprop(lr)
    if name =='sgd':
        optim = SGD(lr)
    if name =='adamax':
        optim = Adamax(lr)
    if name =='nadam':
        optim = Nadam(lr)
    return optim
#Adadelta 1.0  Adagrad 0.01  RMSprop 0.001 SGD 0.01  Adamax 0.002 Nadam 0.002
model.compile(optimizer=optimize(args.rate,args.optname), loss=CrossEntropyLoss(model,10), metrics=[binary_accuracy_with_nan,binary_crossentropy_with_nan,MSE(model)])
#model.summary()

number = int(args.datasize)
with h5py.File('/home/chenxupeng/projects/deepshape/data/new/train_0') as f:
    images_train = f['train_images'][:number]
#with h5py.File('/home/chenxupeng/projects/deepshape/data/new/test_0') as f:
#images_test= f['test_images'][:number]
with h5py.File('/home/chenxupeng/projects/deepshape/data/new/Spitale_2015_invivo_CDS_0.1') as f:
    y_train = f['y_train'][:number]
#y_test = f['y_test'][:number]
y_train = np.concatenate([y_train,y_train],axis = 1)
#y_test= np.concatenate([y_test,y_test],axis = 1)
#y size 256

#callbacks = [model_checkpoint]
model_checkpoint = ModelCheckpoint('output/msdcnn_3.12.hdf5', monitor='binary_accuracy_with_nan', save_best_only=True)
#EarlyStopping(monitor=binary_accuracy_with_nan, patience=10, verbose=0)
def Model(images_train,y_train,cv,seq_counts):
    model.fit(images_train, y_train, batch_size=32, nb_epoch=200,
              verbose=1, shuffle=True,validation_split=0.2,
              callbacks=[model_checkpoint,EarlyStopping(monitor='CrossEntropyLoss', patience=10, verbose=0),TensorBoard(log_dir='/home/chenxupeng/projects/deepshape/output/tensorboard/unet/log_dir')])
    model.save('output/msdcnn_3.12.hdf5')

for i in range(1):
    Model(images_train[:number,:,:,:],y_train[:number,:],i,number)
