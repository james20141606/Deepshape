<<<<<<< HEAD
# coding: utf-8
#! /usr/bin/env python

=======
#! /usr/bin/env python
>>>>>>> 0d1d7cdb80a7b73c9091f3ed002ad5950932669c
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
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser()
parser.add_argument('-n', dest='number')
parser.add_argument('-a', dest='alpha')
parser.add_argument('-e', dest='epoch')
args = parser.parse_args()

model = UNET_128()
optim = Adam()
<<<<<<< HEAD
alpha = int(agrs.alpha)
=======
alpha = int(args.alpha)
>>>>>>> 0d1d7cdb80a7b73c9091f3ed002ad5950932669c
model.compile(optimizer=optim, loss=CrossEntropyLoss(model,alpha), metrics=[binary_accuracy_with_nan,binary_crossentropy_with_nan,MSE(model)])

number = int(args.number)
with h5py.File('/home/chenxupeng/projects/deepshape/data/new/train_1') as f:
    images_train = f['train_images'][:number]
with h5py.File('/home/chenxupeng/projects/deepshape/data/new/test_0') as f:
    images_test= f['test_images'][:number]
with h5py.File('/home/chenxupeng/projects/deepshape/data/new/Spitale_2015_invivo_CDS_0.1') as f:
    y_train = f['y_train'][:number]
    y_test = f['y_test'][:number]
y_train = np.concatenate([y_train,y_train],axis = 1)
y_test= np.concatenate([y_test,y_test],axis = 1)
#y size 256


model_checkpoint = ModelCheckpoint('output/newunet_row_col_mse.hdf5', monitor='binary_accuracy_with_nan', save_best_only=True)

epoch = int(args.epoch)
def Model(images_train,images_test,y_train,y_test_true,count):
<<<<<<< HEAD
    model.fit(images_train, y_train, batch_size=16, nb_epoch=epoch,
              verbose=1, shuffle=True,validation_split=0.2,
              callbacks=[model_checkpoint,EarlyStopping(monitor='CrossEntropyLoss', patience=10, verbose=0),TensorBoard(log_dir='/home/chenxupeng/projects/deepshape/output/tensorboard/unet/log_dir')])
=======
    model.fit(images_train, y_train, batch_size=32, nb_epoch=epoch,
              verbose=1, shuffle=True,validation_split=0.2,
              callbacks=[model_checkpoint,EarlyStopping(monitor='CrossEntropyLoss', patience=30, verbose=0),TensorBoard(log_dir='/home/chenxupeng/projects/deepshape/output/tensorboard/unet/log_dir')])
>>>>>>> 0d1d7cdb80a7b73c9091f3ed002ad5950932669c
    num_test = images_test.shape[0]
    y_test = np.ndarray([num_test,256],dtype=np.float32)
    predict = model.predict([images_test], verbose=0)
    for i in tqdm(range(num_test)):
        y_test[i] = predict[i]
    np.save('/home/chenxupeng/projects/deepshape/output/unet_mse_predict'+str(count)+'.npy', y_test)
    acc = 0.0
    y_test[np.where(y_test >= 0.5)] = 1
    y_test[np.where(y_test < 0.5)] = 0
    for i in range(num_test):
        accu = 0.0
        index = np.where(np.isnan(y_test[i]) ==0)[0]
        count = index.shape[0]
        for j in range(count):
            if y_test[i][index][j] == y_test_true[i][index][j]:
                accu +=1
        accu /=float(count)
        acc +=accu
    acc /=num_test
    print("accuracy : ",acc)
    model.save('output/newunet_row_col_mse.hdf5')
    model.save_weights('output/newunet__row_col_weights_mse.hdf5')
    return acc

for i in range(1):
    acc = Model(images_train[:number,:,:,:],images_test[:number,:,:,:],y_train[:number,:],y_test[:number,:],i)
