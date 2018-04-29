# coding: utf-8
#! /usr/bin/env python
import numpy as np
import argparse, sys, os, errno
from glob import glob
import pandas as pd
import h5py
from tqdm import tqdm
import keras as K
parser = argparse.ArgumentParser()
parser.add_argument('-m', dest='model_path')
parser.add_argument('-s', dest='save_path')
parser.add_argument('-c', dest='count')
parser.add_argument('-p', dest='save_predict')
parser.add_argument('-g', dest='gpu')
parser.add_argument('-i', dest='pictures')
#parser.add_argument('-c', dest='count')
args = parser.parse_args()
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from rnnseq2seq import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import tensorflow as tf
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects


model_path = args.model_path
save_path = args.save_path
count=int(args.count)


#540
ic_shape = {}
with h5py.File('known/known.h5') as f:
    for i in tqdm(range(count)):
        ic_shape[i] = f['known'][f['start'][i]:f['end'][i]]
    icshape = np.array([value for (key,value) in ic_shape.iteritems()])
    name = f['name'][:]


def Model(model_path):
    #prepare for model to predict
    model = UNET_128()
    optim = Adam()
    model.compile(optimizer=optim, loss=CrossEntropyLoss(model,10), metrics=[binary_accuracy_with_nan,binary_crossentropy_with_nan,MSE(model)])
    loss=CrossEntropyLoss(model,10)
    model = load_model(model_path,custom_objects = {"CrossEntropyLoss": loss,\
                       'binary_accuracy_with_nan':binary_accuracy_with_nan,\
                       'binary_crossentropy_with_nan':binary_crossentropy_with_nan,\
                       'MSE':MSE(model)})
    return model
model = Model(model_path)

def calculate_acc(array,true_score):
    '''
        an array of shape (length)*128
        对齐并且求中间每个位置的平均
        return shape*shape+128
        '''
    shape = array.shape[0]
    new = np.ndarray([shape,shape+128])
    for i in range(shape+1):
        new[i] = np.concatenate((np.concatenate((np.zeros(i),array[i])),np.zeros(shape-i)))
    score_vector = np.sum(new,axis = 0)[64:-64].astype('float')  #vector  shape
    #这里要分情况！ 长度 64 128为界限  count_vector不一样
    #64以下 每个位置被算长度次
    if shape <=64:
        count_vector = np.repeat(shape+1,shape+1)
    if shape >=128:
        count_vector = np.concatenate((np.concatenate((np.arange(65,129),np.repeat(128,shape-128))),~np.arange(65,129) +193))
    if shape >64 and shape <128:
        count_vector = np.concatenate((np.concatenate((np.arange(65,shape+1),np.repeat(shape,128-shape))),~np.arange(65,shape+1) +shape+1 +65))
    score = score_vector/count_vector
    for i in range(shape):
        if score[i] <0.5:
            score[i] = 0
        else:
            score[i] = 1
    acc = float(np.where(np.abs(score-true_score) ==0)[0].shape[0])/float(shape)
    return acc
acc= {}
predict_result = {}
with h5py.File(args.pictures) as f:
    for i in tqdm(range(count)):
        images = f[str(i)][:]
        predictval = model.predict(images)
        if int(args.save_predict):
            with h5py.File(save_path+'wholeprediction.hdf5') as m:
                m.create_dataset(str(i),data = predictval)
        acc[i] = calculate_acc(predictval[:,:128], icshape[i])
        with h5py.File(save_path) as t:
            t.create_dataset(str(i),data = acc[i])
acc = np.array([val for (key,val) in acc.iteritems()])
np.savetxt(save_path+'_eachseq.txt',acc)

category = np.ndarray([count]).astype('S')
for i in range(count):
    category[i] = name[i].split('_')[0]
name_list = np.unique(category,)
acc_cate = {}
for j in range(10):
    acc_cate[name_list[j]] = []
    for i in range(count):
        if category[i] ==name_list[j]:
            acc_cate[name_list[j]].append(acc[i])
table = pd.DataFrame([10])
for i in range(10):
    table[i] =  sum(acc_cate[name_list[i]])/len(acc_cate[name_list[i]])
table = table.T
table = table.set_index(name_list)
table.columns = ['2d_model']
#table['2d_model_restrict'] = np.array(pd.read_csv('known/accuracy/acc_1.30_3')['2d_model'])
table['dense'] = ['0.597','0.684','0.628','0.634','0.587','0.524','0.578','0.563','0.555','0.602']
table = table.round(3)
table.to_csv(save_path)
