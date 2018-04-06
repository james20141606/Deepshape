#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
import h5py
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input_file')
parser.add_argument('-o', dest='output_file')
parser.add_argument('-s', dest='window_size')
#parser.add_argument('-c', dest='count')
args = parser.parse_args()

winsize = int(args.window_size)
def convert_to_image(x):
    m = (np.repeat(x, 4, axis=1)[np.newaxis, :, :]*np.tile(x, 4)[:, np.newaxis, :])
    return m

f =  h5py.File(args.input_file, 'r')
num = f['X_train'][:].shape[0]
intval = num/10000
left = num - 10000*intval

for i in tqdm(range(intval+1)):
    if i <intval:
        X_train = f['X_train'][i*10000:(i+1)*10000]
        y_train = f['y_train'][i*10000:(i+1)*10000]
        imgs_train = np.ndarray([10000, winsize, winsize, 16])
        for j in tqdm(range(10000)):
            imgs_train[j] = convert_to_image(X_train[j])
    else:
        X_train = f['X_train'][i*10000:]
        y_train = f['y_train'][i*10000:]
        imgs_train = np.ndarray([left, winsize, winsize, 16])
        for j in tqdm(range(left)):
            imgs_train[j]= convert_to_image(X_train[j])
    with h5py.File(args.output_file+'_'+str(i)) as t:
        t.create_dataset('train_images',data = imgs_train)
        t.create_dataset('y_train', data=y_train)
