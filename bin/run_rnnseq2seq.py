import numpy as np
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', dest='save_path')
parser.add_argument('-i', dest='input_path')
parser.add_argument('-e', dest='epoch')
parser.add_argument('-g', dest='gpu')
parser.add_argument('-model-ind', dest='model_ind')
parser.add_argument('-en-depth', dest='en_depth')
parser.add_argument('-de-depth', dest='de_depth')
parser.add_argument('-dep', dest='dep')
parser.add_argument('-hid-dim', dest='hid_dim')
args = parser.parse_args()
#model_ind = 3, en_depth = 4, de_depth = 5, dep = 4, hid_dim = 10

import sys,os,errno,gc,sys
from glob import glob
import h5py
from tqdm import tqdm
import keras as K
sys.path.append('bin')
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from rnnseq2seq import *
from keras.callbacks import ModelCheckpoint
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def get_data():
    with h5py.File(args.input_path,'r') as f:
        x_train = f['X_train'][:]
        y_train = f['y_train'][:].reshape(-1, 128, 1)
    return x_train, y_train

def data_generator(data, labels, batch_size):
    batches = (data.shape[0] + batch_size - 1)//batch_size
    while(True):
        for i in range(batches):
            X = data[i*batch_size: (i+1)*batch_size]
            Y = labels[i*batch_size: (i+1)*batch_size]
            yield (X, Y)

model = seq2seq_model(args.model_ind,int(args.en_depth),int(args.de_depth),int(args.dep),int(args.hid_dim))

def Model(X_train, Y_train, batch_size=32, num_epochs=int(args.epoch)):
    #model_checkpoint = ModelCheckpoint(args.model_path,
                        #monitor='binary_accuracy_with_nan', save_best_only=True) model_checkpoint,
    #callbacks = [ EarlyStopping(monitor='binary_crossentropy_with_nan',
        #patience=10, verbose=0), TensorBoard(log_dir='deepshape/output/tensorboard/unet/log_dir')]

    model.fit_generator(generator=data_generator(X_train, Y_train, batch_size),
                        steps_per_epoch=(
                            X_train.shape[0] + batch_size - 1) // batch_size,
                        epochs=num_epochs, verbose=1)#, callbacks=callbacks)

    
x_train, y_train = get_data()
Model(x_train, y_train, 32, int(args.epoch))

model.save(args.save_path)
