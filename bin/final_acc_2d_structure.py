import h5py
import os
from tqdm import tqdm
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', dest='model_path')
parser.add_argument('-s', dest='save_path')
parser.add_argument('-g', dest='gpu')
parser.add_argument('-o', dest='opt')  #1 or 0  use fill or not
#parser.add_argument('-c', dest='count')
args = parser.parse_args()
import sys,os,errno,gc
from glob import glob
import pandas as pd
import h5py
from tqdm import tqdm
import keras as K
sys.path.append('bin')
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from second_structure_unet import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


filenames = np.loadtxt('known/ct/filenames.txt',dtype='str')
test_set = np.loadtxt('known/2d/test_set').astype('int')
xtest = {}
ytest = {}
with h5py.File('known/2d/pictures_540') as f:
    for i in range(40):
        xtest[i] = f[filenames[test_set][i]+'/X'][:]
        ytest[i] = f[filenames[test_set][i]+'/y'][:]

def weighthed_binary_ce_loss_nan(alpha):
    #not_nan = tf.logical_not(tf.is_nan(y_true))
    def celoss(y_true, y_pred):
        zero = K.equal(y_true , K.zeros((1,)))
        one = K.equal(y_true , K.ones((1,)))
        weights = np.array([1,alpha])
        y_true_0 = tf.boolean_mask(y_true, zero)
        y_pred_0 = tf.boolean_mask(y_pred, zero)
        y_true_1 = tf.boolean_mask(y_true, one)
        y_pred_1 = tf.boolean_mask(y_pred, one)
        loss_0 = - y_true_0 * K.log(y_pred_0) * K.variable(weights[0])
        loss_1 = - y_true_1 * K.log(y_pred_1) * K.variable(weights[1])
        loss = K.sum(loss_0, -1) +K.sum(loss_1,-1)
        return loss
    return celoss
keras.losses.weighthed_binary_ce_loss_nan = weighthed_binary_ce_loss_nan
model = UNET_128()
model.load_weights('output/2dunetweightedloss_weights.hdf5')
optim = Adam()
model.compile(optimizer=Adam(lr=1e-4),loss=weighthed_binary_ce_loss_nan(731), metrics=['accuracy','MSE'])
loss=weighthed_binary_ce_loss_nan(731)

def get_final(index,opt):
    data = xtest[index]
    a = data.shape[0]
    if a >= 128:
        winsize = 128
    else:
        winsize = 32
    len = int(winsize/2)
    if opt:
        arr = np.zeros([winsize+a,winsize+a,16])
        arr[len:-len,len:-len] = data
        valarr = np.zeros([(a+1)**2,a+winsize,a+winsize])
        for i in tqdm(range(a+1)):
            for j in range(a+1):
                valarr[i*(a+1)+j,i:i+winsize,j:j+winsize] = model.predict(arr[i:i+winsize,j:j+winsize,:].reshape(1,winsize,winsize,16)).reshape(winsize,winsize)
        needarr = valarr[:,len:-len,len:-len]  #[(a+1)**2,a,a]
    else:
        arr = np.zeros([a, a, 16])
        valarr = np.zeros([(a+1-winsize)**2, a,a])
        for i in tqdm(range(a+1-winsize)):
            for j in range(a+1-winsize):
                valarr[i*(a+1-winsize)+j, i:i+winsize, j:j+winsize] = model.predict(arr[i:i+winsize,
                                                                                j:j+winsize, :].reshape(1, winsize, winsize, 16)).reshape(winsize, winsize)
        needarr = valarr[:, len:-len, len:-len]
    countarr = np.zeros([needarr.shape[0],needarr.shape[1],needarr.shape[2]])
    countarr[np.where(needarr !=0)] = 1
    finalarr = np.sum(needarr,axis=0)/np.sum(countarr,axis=0)
    finalarr[finalarr >= 0.5] = 1
    finalarr[finalarr < 0.5] = 0
    del valarr, needarr, countarr
    return finalarr


opt1 = np.setdiff1d(np.arange(5,40), np.array([5, 16, 24, 33]))
opt2 = np.setdiff1d(range(40), np.array([16, 33]))

if int(args.opt):
    save_path = 'output/acc/'+args.save_path+'_fill.hdf5'
    for i in opt1:
        with h5py.File(save_path) as f:  # output/acc/5.1_final_acc_2d.hdf5
            result = get_final(i, int(args.opt))
            f.create_dataset(str(i), data=result)
            del result
else:
    save_path = 'output/acc/'+args.save_path+'_nofill.hdf5'
    for i in opt2:
        with h5py.File(save_path) as f:  # output/acc/5.1_final_acc_2d.hdf5
            result = get_final(i, int(args.opt))
            f.create_dataset(str(i), data=result)
            del result
            print (i)


