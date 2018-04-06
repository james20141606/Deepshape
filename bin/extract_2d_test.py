#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
import h5py
import seaborn as sns
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input_file')
parser.add_argument('-c', dest='count')
args = parser.parse_args()

def convert_to_image(x):
    m = (np.repeat(x, 4, axis=1)[np.newaxis, :, :]*np.tile(x, 4)[:, np.newaxis, :])
    return m


i = int(args.count)


f =  h5py.File(args.input_file, 'r')
num = f['X_test'][:].shape[0]
number = 10**(len(str(num)) -1)
int = num/number
left = num - number*int
if i <int:
    X_test = f['X_test'][i*number:(i+1)*number,:,:]
    imgs_test = np.ndarray([number,128,128,16])
else:
    X_test = f['X_test'][i*number:,:,:]
    imgs_test = np.ndarray([left,128,128,16])
if i <int:
    X_test = f['X_test'][i*number:(i+1)*number,:,:]
    imgs_test = np.ndarray([number,128,128,16])
else:
    X_test = f['X_test'][i*number:,:,:]
    imgs_test = np.ndarray([left,128,128,16])



if i <int:
    for j in tqdm(range(number)):
        imgs_test[j]= convert_to_image(X_test[j])
else:
    for j in tqdm(range(left)):
        imgs_test[j]= convert_to_image(X_test[j])

with h5py.File('/home/chenxupeng/projects/deepshape/data/new/test_'+args.count) as t:
    t.create_dataset('test_images',data = imgs_test)
