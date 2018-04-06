#! /usr/bin/env python
# coding: utf-8
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
import seaborn as sns
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from unet_128_model_row_column import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
set_session(tf.Session(config=config))
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='index')
parser.add_argument('-c', dest='count')
args = parser.parse_args()

i = int(args.index)
count = int(args.count)

def prepare_data(numbers):
    '''
        use different seqs
        '''
    seqs = {}
    with h5py.File('data/new/Spitale_2015_invivo_CDS_0.1') as f:
        for i in range(numbers):
            seqs[i] = f['X_train'][128*i]
    return seqs

def convert_to_image(x):
    m = (np.repeat(x, 4, axis=1)[np.newaxis, :, :]*np.tile(x, 4)[:, np.newaxis, :])
    return m
def Model():
    #为转成十六通道的图片预测shape做准备
    model = UNET_128()
    optim = Adam()
    model.compile(optimizer=optim, loss=CrossEntropyLoss(model,10), metrics=[binary_accuracy_with_nan,binary_crossentropy_with_nan,MSE(model)])
    loss=CrossEntropyLoss(model,10)
    model = load_model('output/unet_for_mutation_map.hdf5',custom_objects = {"CrossEntropyLoss": loss,\
                       'binary_accuracy_with_nan':binary_accuracy_with_nan,\
                       'binary_crossentropy_with_nan':binary_crossentropy_with_nan,\
                       'MSE':MSE(model)})
    return model
def generate_mutation_map(seq):
    '''seq shape: 128*4
        将一条序列每个位点突变
        return 384*128*4
        '''
    length = 128
    #需要生成的 1的位置
    mutated = np.ndarray([length*3,length,4]).astype('int')
    for j in range(length):
        position = np.setdiff1d(np.array([0,1,2,3]),np.where(seq[j]==1)[0][0])
        for i in range(3):
            a = np.zeros(4).astype('int')
            array = np.copy(seq)
            a[position[i]] = 1
            array[j] = a
            mutated[j*3+i] = array
    images_mutated = np.ndarray([length*3,length,length,16]).astype('int')
    for i in range(length*3):
        images_mutated[i] = convert_to_image(mutated[i])
    images_origin = convert_to_image(seq)
    model = Model()
    predict_origin = model.predict(images_origin.reshape(1,128,128,16))[0][:128]
    predict_mutated = model.predict(images_mutated)[:,:128]
    mutation_map = np.ndarray([128*3,128])
    for i in range(128*3):
        mutation_map[i] =  predict_origin - predict_mutated[i]
    return mutation_map

def mutiple_mutation_maps(seqs,numbers):
    '''
        output many seqs mutation maps
        '''
    mutation_map = {}
    for i in tqdm(range(numbers)):
        mutation_map[i] = generate_mutation_map(seqs[i])
    mutation_map_ = np.array([val for (key,val) in mutation_map.iteritems()])
    mutation_map_ = mutation_map_.astype('float')
    return mutation_map_

def convert_to_cube(map):
    cube_map = np.ndarray([384,384])
    for i in range(128):
        cube_map[:,3*i] = map[:,i]
        cube_map[:,3*i+1] = map[:,i]
        cube_map[:,3*i+2] = map[:,i]
    return cube_map
def multi_cube_map(numbers):
    '''generate different seqs' cube array '''
    cube_map = {}
    for i in range(100):
        cube_map[i] = convert_to_cube(mutation_map_[i])
    return cube_map

def generate_boxplot(mutationmap):
    new_map = np.ndarray([384,256])
    for i in range(128):
        new_map[3*i] = np.concatenate((np.concatenate((np.zeros(128-i),mutationmap[3*i])),np.zeros(i)))
        new_map[3*i+1] = np.concatenate((np.concatenate((np.zeros(128-i),mutationmap[3*i+1])),np.zeros(i)))
        new_map[3*i+2] = np.concatenate((np.concatenate((np.zeros(128-i),mutationmap[3*i+2])),np.zeros(i)))
    return new_map

def generate_mutiple_boxplot(mutataionmaps,numbers):
    '''
        numbers: 序列数 map数
        mutationmaps:number个数个mutation map在一起
        '''
    new_map = np.ndarray([numbers,384,256])
    for i in range(numbers):
        new_map[i] = generate_boxplot(mutationmaps[i])
    return new_map

def main():
    seqs = prepare_data(100)
    seqs = np.array([val for (key,val) in seqs.iteritems()])
    seqs = seqs[i*count:(i+1)*count]

    muti_mut_maps = mutiple_mutation_maps(seqs,count)

    with h5py.File('mutation/mutation_maps_100_no_overlap') as f:
        f.create_dataset('maps'+str(i),data = muti_mut_maps)

if __name__ == "__main__":
    main()



