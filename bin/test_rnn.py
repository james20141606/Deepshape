#! /usr/bin/env python

import os, sys, argparse
# import keras
execfile(os.path.join(os.path.dirname(__file__), 'import_keras.py'))

def save_dataset(dataset, outfile):
    """Save a dataset to an HDF5 file
    dataset: a dict of numpy arrays
    """
    import h5py
    f = h5py.File(outfile, 'w')
    for key in dataset:
        f.create_dataset(key, data=dataset[key])
    f.close()

def load_dataset(infile):
    """Load a dataset from an HDF5 file
    Returns a dict of numpy arrays with keys as variable names.
    """
    import h5py
    f = h5py.File(infile, 'r')
    dataset = {}
    for key in f.keys():
    	dataset[key] = f[key][:]
    f.close()
    return dataset

def binary_crossentropy_rnn(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true))

def create_dataset_rnn1(length=100, n_samples=2000, window_size=5, offset=None, seed=None, **kwargs):
    import numpy as np
    from scipy.signal import convolve

    conv_size = length - window_size + 1
    if offset is None:
        offset = window_size - 1
    else:
        offset = np.clip(offset, -conv_size + 1, length - 1)
    if seed:
        np.random.seed(seed)
    X = np.random.uniform(0.0, 1.0, size=(n_samples, length))
    filter = np.ones(window_size)

    y = np.zeros((n_samples, length), dtype='int8')
    for i in range(n_samples):
        y_all = convolve(X[i, :], filter, mode='valid') > float(window_size)/2
        if offset < 0:
            y[i, :(conv_size + offset)] = y_all[-offset:]
        elif offset <= (window_size - 1):
            y[i, offset:(offset + conv_size)] = y_all
        else:
            y[i, offset:] = y_all[:(length - offset)]
    #y_valid = convolve(X, filter, mode='valid')
    #y = np.zeros((n_samples, length))
    #y[:, (window_size-1):] = y_valid
    X = np.expand_dims(X, axis=2)
    y = np.expand_dims(y, axis=2)
    return (X, y)

def train_model(X, y, model_file, batch_size=25, nepoch=20, **kwargs):
    window_size = X.shape[1]
    n_samples = X.shape[0]
    X_train = X[:n_samples/2, :]
    y_train = y[:n_samples/2, :]
    X_test  = X[n_samples/2:, :]
    y_test  = y[n_samples/2:, :]

    _defered_import()
    from keras.layers.recurrent import SimpleRNN
    from keras.optimizers import RMSprop

    # load model
    with open(model_file, 'r') as f:
        exec compile(f.read(), model_file, 'exec')

    optimizer = RMSprop(lr=0.0005)
    model.compile(optimizer=optimizer,
	              loss=binary_crossentropy_rnn,
              	  metrics=['binary_accuracy'])
    model.summary()

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
        batch_size=batch_size, nb_epoch=nepoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, choices=('train', 'create_dataset'))
    parser.add_argument('--dataset', type=str, required=False, choices=('rnn1'),
        help='the name of the dataset to create')
    parser.add_argument('-o', '--outfile', type=str, required=False,
        help='the output file name')
    parser.add_argument('-i', '--infile', type=str, required=False,
        help='the input file name')
    parser.add_argument('--batch-size', type=int, required=False,
        help='batch size for training')
    parser.add_argument('--nepoch', type=int, required=False,
        help='number of epochs for training')
    parser.add_argument('--nsamples', type=int, required=False,
        help='number of samples to generate')
    parser.add_argument('--window-size', type=int, required=False,
        help='window size for convolution in creating datasets')
    parser.add_argument('--length', type=int, required=False,
        help='length of each sample')
    parser.add_argument('--model', type=str, required=False,
        help='a python script that build a Keras model (named model)')
    parser.add_argument('--offset', type=int, required=False,
        help='offset for creating the rnn dataset')
    parser.add_argument('--seed', type=int, required=False,
        help='random seed for creating datasets')
    args = parser.parse_args()
    args_valid = {k:v for k, v in vars(args).iteritems() if v is not None}

    if args.command == 'train':
        dataset = load_dataset(args.infile)
        train_model(dataset['X'], dataset['y'], args.model, **args_valid)
    elif args.command == 'create_dataset':
        if args.dataset == 'rnn1':
            X, y = create_dataset_rnn1(**args_valid)
            save_dataset({'X': X, 'y': y}, args.outfile)
        else:
            raise ValueError('invalid dataset name: {}'.format(args.dataset))
