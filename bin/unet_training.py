#! /usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import argparse, sys, os, errno
import os
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from tqdm import tqdm
import keras as K
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping


np.random.seed(1234)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))


from keras.callbacks import EarlyStopping
from train_unet import *
model = get_unet()
#model_checkpoint = ModelCheckpoint('output/unet2.hdf5', monitor='loss', save_best_only=True)
def Model(images_train,images_test,masks_train,masks_test_true):
    model.fit(images_train, masks_train, batch_size=16, nb_epoch=50,
              verbose=1, shuffle=True,validation_split=0.2,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=0),TensorBoard(log_dir='/home/chenxupeng/projects/deepshape/output/tensorboard/unet/log_dir')])
    num_test = images_test.shape[0]
    masks_test = np.ndarray([num_test,16,256,256],dtype=np.float32)
    predict = model.predict([images_test], verbose=0)
    for i in tqdm(range(num_test)):
        masks_test[i] = predict[i]
    np.save('/home/chenxupeng/projects/deepshape/output/maskpredict.npy', masks_test)
    mean = 0.0
    for i in range(num_test):
        mean+=dice_coef_np(masks_test_true[i,0], masks_test[i,0])
    mean/=num_test
    print("Mean Dice Coeff : ",mean)
    return mean


images_train={}
images_test={}
masks_train={}
masks_test_true={}
for i in range(10):
    f =  h5py.File('/home/chenxupeng/projects/deepshape/output/images_cv')
    images_train[i] = f['train_images_'+str(i)][:,:,:,:]
    images_test[i] = f['test_images_'+str(i)][:,:,:,:]
    masks_train[i] = f['train_masks_'+str(i)][:,:,:,:]
    masks_test_true[i] =f['test_masks_'+str(i)][:,:,:,:]

mean = {}
for i in tqdm(range(10)):
    mean[i] = Model(images_train[i][:,:,:,:],images_test[i][:,:,:,:],masks_train[i][:,:,:,:],masks_test_true[i][:,:,:,:])

model.save('/home/chenxupeng/projects/deepshape/output/unet.hdf5')

