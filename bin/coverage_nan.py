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
parser.add_argument('-o', dest='output')
parser.add_argument('-v', dest='coverage')
args = parser.parse_args()

with h5py.File('data/new/Spitale_2015_invivo_CDS_percentile_25') as f:
    y_test = f['y_test'][:]
    y_train = f['y_train'][:]

shape0 = y_train.shape[0]
shape1= y_test.shape[0]

def convert_nan(y,count):
    y_f = y.flatten()
    count = y_f.shape[0]
    index_notnan = np.where(np.isnan(y_f)==0)
    count_convert = int(count*0.1*count)
    np.random.seed(1111)
    np.random.shuffle(index_notnan[0])
    pick_index = index_notnan[0][:count_convert]
    y_f[pick_index] = np.nan
    y_ = y_f.reshape(-1,128)
    return y_

for i in range(4):
    count = i+1
    y_train_ = np.ndarray([shape0,128])
    y_test_ = np.ndarray([shape1,128])
    y_train_ = convert_nan(y_train,count)
    y_test_ = convert_nan(y_test,count)
    with h5py.File('new/y_Spitale_2015_invivo_CDS_percentile_25_'+str(count*0.1+0.5)) as f:
        f.create_dataset('y_train',data = y_train_)
        f.create_dataset('y_test',data = y_test_)

with h5py.File('new/y_Spitale_2015_invivo_CDS_percentile_25_0.5') as f:
    f.create_dataset('y_train',data = y_train)
    f.create_dataset('y_test',data = y_test)

