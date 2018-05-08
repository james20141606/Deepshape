from tqdm import tqdm
import argparse
import numpy as np
import sys,os,errno,gc
import numba
from glob import glob
import pandas as pd
import h5py
import keras as K
sys.path.append('bin')
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from second_structure_unet import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('-m', dest='model_path')
parser.add_argument('-s', dest='save_path')
parser.add_argument('-g', dest='gpu')
parser.add_argument('-o', dest='opt')
parser.add_argument('-f', dest='func')   #1 or 0  use fill or not
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


filenames = np.loadtxt('known/ct/filenames.txt',dtype='str')
test_set = np.loadtxt('known/2d/test_set').astype('int')
xtest = {}
ytest = {}
with h5py.File('known/2d/pictures_540','r') as f:
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
        try:
            arr = np.zeros([winsize+a,winsize+a,16])
            arr[len:-len,len:-len] = data
            valarr = np.zeros([(a+1)**2,a+winsize,a+winsize])
            for i in tqdm(range(a+1)):
                for j in range(a+1):
                    valarr[i*(a+1)+j,i:i+winsize,j:j+winsize] = model.predict(arr[i:i+winsize,j:j+winsize,:].reshape(1,winsize,winsize,16)).reshape(winsize,winsize)
            needarr = valarr[:,len:-len,len:-len]  #[(a+1)**2,a,a]
            countarr = np.zeros([needarr.shape[0],needarr.shape[1],needarr.shape[2]])
            countarr[np.where(needarr !=0)] = 1
            finalarr = np.sum(needarr,axis=0)/np.sum(countarr,axis=0)
            finalarr[finalarr >= 0.5] = 1
            finalarr[finalarr < 0.5] = 0
            del valarr, needarr, countarr
            return finalarr
        except MemoryError:
            print ('\n'+'skip' +str(index))
            with open('output/acc/fill_memoryerr.txt','a') as f:
                f.write(str(index)+'\n')
            return 0
    else:
        try:
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
        except MemoryError:
            print ('\n'+'skip' +str(index))
            with open('output/acc/nofill_memoryerr.txt','a') as f:
                f.write(str(index)+'\n')
            return 0
    

def get_final_slice(index,opt):
    '''
    save slice need arr
    index/i/j
    output/acc/slices_fill.hdf5
    '''
    data = xtest[index]
    a = data.shape[0]
    if a >= 128:
        winsize = 128
    else:
        winsize = 32
    len = int(winsize/2)
    if opt:
        save_path = 'output/acc/slices_fill.hdf5'
        with h5py.File(save_path) as f:
            if np.where(np.array(list(f.keys()))==str(index))[0].shape[0] ==0:
                arr = np.zeros([winsize+a,winsize+a,16])
                arr[len:-len,len:-len] = data
                for i in tqdm(range(a+1)):
                    for j in range(a+1):
                        valarr = np.zeros([a+winsize,a+winsize])
                        valarr[i:i+winsize,j:j+winsize] = model.predict(arr[i:i+winsize,j:j+winsize,:].reshape(1,winsize,winsize,16)).reshape(winsize,winsize)
                        needarr = valarr[len:-len, len:-len]
                        f.create_dataset(str(index)+'/'+str(i)+','+str(j),data=needarr)
            else:
                print ('already have slice '+str(index))
    else:
        save_path = 'output/acc/slices_nofill.hdf5'
        with h5py.File(save_path) as f:
            if np.where(np.array(list(f.keys()))==str(index))[0].shape[0] ==0:
                arr = np.zeros([a, a, 16])
                for i in tqdm(range(a+1-winsize)):
                    for j in range(a+1-winsize):
                        valarr = np.zeros([a,a])
                        valarr[i*(a+1-winsize)+j, i:i+winsize, j:j+winsize] = model.predict(arr[i:i+winsize,
                                                                                        j:j+winsize, :].reshape(1, winsize, winsize, 16)).reshape(winsize, winsize)
                        needarr = valarr[len:-len, len:-len]
                        f.create_dataset(str(index)+'/'+str(i)+','+str(j),data=needarr)
            else:
                print ('already have slice '+str(index))


#opt1 = np.array([5, 16, 24, 33])
#opt2 = np.array([16, 33])


def final_run():
    if int(args.opt):
        save_path = 'output/acc/'+args.save_path+'_fill.hdf5'
        for i in tqdm(np.arange(0,40)):#opt1:
            with h5py.File(save_path) as f:  # output/acc/5.1_final_acc_2d.hdf5
                if np.where(np.array(list(f.keys()))==str(i))[0].shape[0] ==0:
                    result = get_final(i, int(args.opt))
                    if result !=0:
                        f.create_dataset(str(i), data=result)
                        del result
                    else:
                        print (result)
                else:
                    print ('skip '+str(i)+' because of memory error, save ' str(i))
    else:
        save_path = 'output/acc/'+args.save_path+'_nofill.hdf5'
        for i in tqdm(range(40)):#opt2:
            with h5py.File(save_path) as f:  # output/acc/5.1_final_acc_2d.hdf5
                if np.where(np.array(list(f.keys()))==str(i))[0].shape[0] ==0:
                    print (f.keys())
                    print ('current index'+str(i))
                    result = get_final(i, int(args.opt))
                    if result !=0:
                        f.create_dataset(str(i), data=result)
                        del result
                    else:
                        print ('skip '+str(i)+' because of memory error, save ' str(i))
                else:
                    print ("already done " +str(i))
                    print (f.keys())

if __name__ == '__main__':

    if args.func == 'whole':
        final_run()
    elif args.func == 'slice':
        with h5py.File('output/acc/fill_memoryerr.hdf5'):
            opt1 = f['index'][:].astype('int')
        with h5py.File('output/acc/nofill_memoryerr.hdf5'):
            opt2 = f['index'][:].astype('int')
        if int(args.opt):
            for i in opt1:
                get_final_slice(i,1)
        else:
            for i in opt2:
                get_final_slice(i,0)


