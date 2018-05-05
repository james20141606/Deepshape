#! /usr/bin/env python
from __future__ import print_function
from six.moves import range, xrange

import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
import numpy as np
import numba
import h5py

alphabet = 'ATCG'

def detect_model_format(filename):
    with open(filename, 'rb') as f:
        magic = f.read(4)
        if magic == b'\x89HDF':
            return 'keras'
        else:
            return 'sklearn'
            
def open_hdf5_group(path, mode='r'):
    pos = path.find(':')
    if pos >= 0:
        filename = path[:pos]
        if pos < len(path):
            group = path[(pos + 1):]
        else:
            group = '/'
        return h5py.File(filename, mode)[group]
    else:
        return h5py.File(path, mode)

def get_scorer(name):
    """Returns a function that accept at least two parameters: y_true, y_pred
    """
    import sklearn.metrics
    import scipy.stats
    # See http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    scorers = {
        'accuracy': sklearn.metrics.accuracy_score,
        'average_precision': sklearn.metrics.average_precision_score,
        'f1': sklearn.metrics.f1_score,
        'precision': sklearn.metrics.precision_score,
        'recall': sklearn.metrics.recall_score,
        'roc_auc': sklearn.metrics.roc_auc_score,
        'sensitivity': sklearn.metrics.recall_score,
        'ppv': sklearn.metrics.precision_score,
        'r2': sklearn.metrics.r2_score,
        'mean_squared_error': sklearn.metrics.mean_squared_error,
        'pearson_r': lambda y_true, y_pred: scipy.stats.pearsonr(y_true, y_pred)[0],
        'pearson_p': lambda y_true, y_pred: scipy.stats.pearsonr(y_true, y_pred)[1],
        'spearman_r': lambda y_true, y_pred: scipy.stats.spearmanr(y_true, y_pred)[0],
        'spearman_p': lambda y_true, y_pred: scipy.stats.spearmanr(y_true, y_pred)[1]
        }
    return scorers[name]

def get_scorer_type(name):
    scorer_types = {'accuracy': 'binary',
        'average_precision': 'binary',
        'f1': 'binary',
        'precision': 'binary',
        'recall': 'binary',
        'sensitivity': 'binary',
        'ppv': 'binary',
        'roc_auc': 'continuous',
        'r2': 'continuous',
        'mean_squared_error': 'continuous',
        'pearson_r': 'continuous',
        'pearson_p': 'continuous',
        'spearman_r': 'continuous',
        'spearman_p': 'continuous'
        }
    return scorer_types[name]

@numba.jit
def split_windows_valid(x, window_size, stride):
    out_length = int(np.floor(float(x.shape[0] - window_size)//stride)) + 1
    if out_length < 0:
        out_length = 0
    windows = np.empty((out_length, window_size), dtype=x.dtype)
    i_x = 0
    for i_w in range(out_length):
        for j in range(window_size):
            windows[i_w, j] = x[i_x + j]
        i_x += stride
    return windows

@numba.jit
def split_windows_same(x, window_size, stride, padded_value=None, offset=0):
    L = x.shape[0]
    out_length = int(np.floor(float(L) - 1)//stride) + 1
    windows = np.full((out_length, window_size), padded_value, dtype=x.dtype)
    # left and right padding length
    i_x = 0
    for i_w in range(out_length):
        for j in range(window_size):
            x_pos = i_x + j - offset
            if 0 <= x_pos < L:
                windows[i_w, j] = x[x_pos]
        i_x += stride
    return windows
            
def sequences_to_windows(sequences, window_size=10, stride=1, mode='valid'):
    '''Convert a list of sequences to short fragments of fixed length
    Args:
        sequences: a list of sequences
        mode: similar to convolution mode: valid, same
    Returns: windows, window_starts, window_ends
    '''
    assert (stride > 0) and (window_size > 0)
    windows = []
    if mode == 'valid':
        for sequence in sequences:
            windows.append(split_windows_valid(sequence, window_size, stride))
    elif mode == 'same':
        for sequence in sequences:
            windows.append(split_windows_same(sequence, window_size, stride, 'N'))
    lengths = np.asarray([a.shape[0] for a in windows])
    ends = np.cumsum(lengths)
    starts = ends - lengths
    windows = np.concatenate([onehot_encode(x) for x in windows], axis=0)
    return windows, starts, ends

def onehot_encode(x, alphabet='ATCG'):
    '''
    '''
    alphabet = np.frombuffer(bytearray(alphabet, encoding='ascii'), dtype='S1')
    x_shape = list(x.shape)
    encoded = (x.reshape(x_shape + [1]) == alphabet.reshape([1]*len(x_shape) + [-1])).astype(np.int32)
    return encoded

def train(args):
    import numpy as np
    import keras
    import h5py
    import models
    from ioutils import prepare_output_file, make_dir

    logger.info('load training data: ' + args.input_file)
    fin = h5py.File(args.input_file, 'r')
    X_train = fin[args.xname][:]
    y_train = fin[args.yname][:]
    fin.close()

    valid_data = None
    if args.valid_file:
        logger.info('load validation data: ' + args.valid_file)
        fin = h5py.File(args.valid_file, 'r')
        X_valid = fin[args.valid_xname][:]
        y_valid = fin[args.valid_yname][:]
        fin.close()
        valid_data = (X_valid, y_valid)

    window_size = X_train.shape[1]
    model = getattr(models, args.model)(window_size)
    if args.regression:
        loss = 'mean_squared_error'
        metrics = ['mean_squared_error']
    else:
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    model.compile(optimizer='Adam',
                loss=loss,
                metrics=metrics)
    model.summary()

    callbacks = []
    if args.tensorboard_log_dir:
        from keras.callbacks import TensorBoard
        callbacks = [TensorBoard(log_dir=args.tensorboard_log_dir)]
    else:
        callbacks = []
    if args.keras_log is not None:
        logger.info('open CSV log file: {}'.format(args.keras_log))
        make_dir(os.path.dirname(args.keras_log))
        callbacks.append(keras.callbacks.CSVLogger(args.keras_log))

    logger.info('train model')
    model.fit(X_train, y_train,
        batch_size=args.batch_size, epochs=args.epochs,
        callbacks=callbacks, verbose=args.keras_verbose,
        validation_data=valid_data)
    logger.info('save model: {}'.format(args.model_file))
    prepare_output_file(args.model_file)
    model.save(args.model_file)

def evaluate(args):
    import numpy as np
    import keras
    import h5py
    import models
    import six.moves.cPickle as pickle
    from ioutils import prepare_output_file, make_dir
            
    logger.info('load model: {}'.format(args.model_file))
    model_format = detect_model_format(args.model_file)
    logger.info('detected model format: ' + model_format)
    if model_format == 'keras':
        model = keras.models.load_model(args.model_file)
    elif model_format == 'sklearn':
        with open(args.model_file, 'r') as f:
            model = pickle.load(f)

    logger.info('load data: {}'.format(args.input_file))
    fin = h5py.File(args.input_file, 'r')
    X_test = fin[args.xname][:]
    y_test = fin[args.yname][:]
    fin.close()

    logger.info('run the model')
    if model_format == 'keras':
        y_pred = model.predict(X_test, batch_size=args.batch_size)
    elif model_format == 'sklearn':
        y_pred = model.predict(X_test)

    y_pred = np.squeeze(y_pred)
    if args.swap_labels:
        logger.info('swap labels')
        y_pred = 1 - y_pred
    y_pred_labels = (y_pred >= args.cutoff).astype('int32')

    # ingore NaNs in y_test
    y_test = y_test.flatten()
    y_pred = y_pred.flatten()
    y_pred_labels = y_pred_labels.flatten()
    not_nan_mask = np.logical_not(np.isnan(y_test))
    y_test = y_test[not_nan_mask]
    y_pred = y_pred[not_nan_mask]
    y_pred_labels = y_pred_labels[not_nan_mask]

    scores = {}
    for metric in args.metrics:
        # y_pred is an array of continous scores
        scorer = get_scorer(metric)
        if metric == 'roc_auc':
            scores[metric] = scorer(y_test, y_pred)
        else:
            scores[metric] = scorer(y_test, y_pred_labels)
        logger.info('metric {} = {}'.format(metric, scores[metric]))
    if args.output_file is not None:
        logger.info('save file: {}'.format(args.output_file))
        prepare_output_file(args.output_file)
        fout = h5py.File(args.output_file, 'w')
        fout.create_dataset('y_true', data=y_test)
        fout.create_dataset('y_pred', data=y_pred)
        fout.create_dataset('y_pred_labels', data=y_pred_labels)
        grp = fout.create_group('metrics')
        for metric in args.metrics:
            grp.create_dataset(metric, data=scores[metric])
        fout.close()
        
def evaluate_structures(args):
    import numpy as np
    import h5py
    from formats import read_dot
    from sklearn.metrics import roc_auc_score, accuracy_score

    logger.info('read input dot file: ' + args.dot_file)
    structures = {}
    for name, seq, structure in read_dot(args.dot_file):
        structures[name] = (np.frombuffer(bytearray(structure, encoding='ascii'), dtype='S1') != b'.').astype(np.int32)
    
    logger.info('read input prediction file: ' + args.pred_file)
    y_preds = {}
    with h5py.File(args.pred_file, 'r') as f:
        for name in f.keys():
            y_preds[name] = f[name][:]
    if args.swap_labels:
        logger.info('swap structure labels')
        for name in structures:
            structures[name] = 1 - structures[name]
    roc_auc = {}
    accuracy = {}
    for name in y_preds:
        roc_auc[name] = roc_auc_score(structures[name], y_preds[name])
        accuracy[name] = accuracy_score(structures[name], (y_preds[name] >= 0.5).astype(np.int32))
    logger.info('create output file: ' + args.output_file)
    with open(args.output_file, 'w') as f:
        f.write('name\troc_auc\taccuracy\n')
        for name in y_preds:
            f.write('{0}\t{1}\t{2}\n'.format(name, roc_auc[name], accuracy[name]))


def predict(args):
    import numpy as np
    import keras
    import h5py
    import models
    from tqdm import tqdm
    import six.moves.cPickle as pickle
    from ioutils import prepare_output_file, make_dir
    from formats import read_fasta

    logger.info('load model: {}'.format(args.model_file))
    model_format = detect_model_format(args.model_file)
    logger.info('detected model format: ' + model_format)
    if model_format == 'keras':
        model = keras.models.load_model(args.model_file)
        window_size = model.input.shape[1].value
    elif model_format == 'sklearn':
        with open(args.model_file, 'r') as f:
            model = pickle.load(f)

    # default offset
    if args.offset is None:
        offset = int(window_size)//2
    logger.info('load data: {}'.format(args.input_file))
    if args.format == 'fasta':
        names = []
        logger.info('create output file: ' + args.output_file)
        fout = h5py.File(args.output_file, 'w')
        for name, sequence in tqdm(read_fasta(args.input_file), unit='transcript'):
            names.append(name)
            sequence = np.frombuffer(bytearray(sequence, encoding='ascii'), dtype='S1')
            windows = split_windows_same(sequence, window_size, 1, offset=offset)
            X = onehot_encode(windows)
            y_pred = model.predict(X, batch_size=args.batch_size)
            y_pred = np.squeeze(y_pred)
            if args.swap_labels:
                logger.info('swap labels')
                y_pred = 1 - y_pred
            fout.create_dataset(name, data=y_pred)
        fout.close()
    else:
        raise ValueError('unknown input format: ' + args.format)
    
def create_single_point_dataset(sequences, targets, names, offset,
        window_size=10, stride=1, mode='same'):
    Xs = []
    ys = []
    for name in names:
        y = targets[name]
        sequence = split_windows_same(sequences[name], window_size, stride, offset=offset, padded_value='N')
        X = onehot_encode(sequence)
        notnan_mask = ~np.isnan(y)
        ys.append(y[notnan_mask])
        Xs.append(X[notnan_mask])
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys)
    return Xs, ys

def create_dense_dataset(sequences, targets, names, offset,
        window_size=10, stride=1, mode='same'):
    Xs = []
    ys = []
    for name in names:
        y = targets[name]
        sequence = split_windows_same(sequences[name], window_size, stride, offset=offset, padded_value='N')
        y = split_windows_same(y[name], window_size, stride, offset=offset, padded_value=np.nan)
        X = onehot_encode(sequence)
        notnan_mask = np.any(~np.isnan(y), axis=1)
        ys.append(y[notnan_mask])
        Xs.append(X[notnan_mask])
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys)
    return Xs, ys

def balance_dataset(X, y, method='subsample_minor'):
    indices_0 = np.nonzero(y == 0)[0]
    indices_1 = np.nonzero(y == 1)[0]
    if indices_0.shape[0] > indices_1.shape[0]:
        subsample_indices = np.random.choice(indices_0, size=indices_1.shape[0], replace=False)
        subsample_indices = np.concatenate([subsample_indices, indices_1])
    else:
        subsample_indices = np.random.choice(indices_1, size=indices_0.shape[0], replace=False)
        subsample_indices = np.concatenate([subsample_indices, indices_0])
    np.random.shuffle(subsample_indices)
    X = X[subsample_indices]
    y = y[subsample_indices]
    return X, y

def create_dataset(args):
    from tqdm import tqdm
    from formats import read_fasta
    from ioutils import prepare_output_file

    c = args.input_file.split(':')
    input_file = c[0]
    dataset = c[1] if len(c) > 1 else '/'
    logger.info('read input file: ' + input_file)
    g_input = open_hdf5_group(args.input_file, 'r')
    names = np.asarray(list(g_input.keys()))
    reactivities = {name:g_input[name][:] for name in names}

    logger.info('read sequence file: ' + args.sequence_file)
    sequences = {name:np.frombuffer(bytearray(seq, encoding='ascii'), dtype='S1') for name, seq in read_fasta(args.sequence_file)}

    if args.offset is None:
        offset = int(args.window_size)//2
    else:
        offset = args.offset

    if args.cv_split_file is not None:
        cv_split = open_hdf5_group(args.cv_split_file, 'r')
        train_index = cv_split['train'][:]
        test_index = cv_split['test'][:]
        names_train = names[train_index]
        names_test = names[test_index]
        X_train, y_train = create_single_point_dataset(sequences, reactivities, names_train, offset,
            args.window_size, args.stride)
        X_test, y_test = create_single_point_dataset(sequences, reactivities, names_test, offset,
            args.window_size, args.stride)
        if args.balanced:
            logger.info('create balanced dataset')
            X_train, y_train = balance_dataset(X_train, y_train)
            logger.info('number of training samples: {}'.format(y_train.shape[0]))
            X_test, y_test = balance_dataset(X_test, y_test)
            logger.info('number of test samples: {}'.format(y_test.shape[0]))

        logger.info('create output file: ' + args.output_file)
        prepare_output_file(args.output_file)
        with h5py.File(args.output_file, 'w') as fout:
            fout.create_dataset('names_train', data=names_train.astype('S'))
            fout.create_dataset('X_train', data=X_train, compression=True)
            fout.create_dataset('y_train', data=y_train, compression=True)
            fout.create_dataset('names_test', data=names_test.astype('S'))
            fout.create_dataset('X_test',  data=X_test, compression=True)
            fout.create_dataset('y_test',  data=y_test, compression=True)
            fout.create_dataset('offset', data=offset)
    else:
        X, y = create_single_point_dataset(sequences, reactivities, names, offset,
            args.window_size, args.stride)
        logger.info('create output file: ' + args.output_file)
        prepare_output_file(args.output_file)
        with h5py.File(args.output_file, 'w') as fout:
            fout.create_dataset('names', data=names.astype('S'))
            fout.create_dataset('X', data=X, compression=True)
            fout.create_dataset('y', data=y, compression=True)

def cv_split(args):
    import numpy as np
    import h5py
    from sklearn.model_selection import KFold

    logger.info('read input file: ' + args.input_file)
    g_input = open_hdf5_group(args.input_file, 'r')
    n_samples = len(g_input.keys())
    logger.info('number of samples: {}'.format(n_samples))
    kfold = KFold(n_splits=args.k, shuffle=True, random_state=args.random_state)
    with h5py.File(args.output_file, 'w') as fout:
        i = 0
        for train_index, test_index in kfold.split(X=np.arange(n_samples).reshape((-1, 1))):
            g = fout.create_group(str(i))
            g.create_dataset('train', data=train_index)
            g.create_dataset('test', data=test_index)
            i += 1

@numba.jit
def binarize_by_cutoffs(x, lower, upper):
    for i in range(x.shape[0]):
        if np.isnan(x[i]):
            x[i] = np.nan
        elif x[i] <= lower:
            x[i] = 0
        elif x[i] >= upper:
            x[i] = 1
        else:
            x[i] = np.nan

def binarize(args):
    import numpy as np
    import h5py

    logger.info('read input file: ' + args.input_file)
    data = {}
    g_input = open_hdf5_group(args.input_file, 'r')
    for name in g_input.keys():
        data[name] = g_input[name][:]
    if args.method.startswith('percentile'):
        method, method_params = args.method.split(':')
        method_params = method_params.split(',')
        lower, upper = int(method_params[0]), int(method_params[1])
        logger.info('binarize using percentiles: {}-{}'.format(lower, upper))
        data_concat = np.concatenate(list(data.values()))
        data_concat = data_concat[~np.isnan(data_concat)]
        lower_cutoff = np.percentile(data_concat, lower)
        upper_cutoff = np.percentile(data_concat, upper)
        logger.info('percentile cutoffs: {}-{}'.format(lower_cutoff, upper_cutoff))
        for name in data:
            binarize_by_cutoffs(data[name], lower_cutoff, upper_cutoff)
    else:
        raise ValueError('unknown method: ' + args.method)
    logger.info('create output file: ' + args.output_file)
    with h5py.File(args.output_file, 'w') as fout:
        for name in data:
            fout.create_dataset(name, data=data[name])

def summarize_metrics(args):
    import numpy as np
    import h5py
    import pandas as pd
    from tqdm import tqdm
    from ioutils import open_file_or_stdout

    def parse_filename(filename):
        d = {}
        keymap = {'d': 'dataset', 'w': 'window_size', 'b': 'binarization_method', 'm': 'model', 'i': 'cv_index'}
        for v in filename.split(','):
            c = v.split('=')
            if len(c) == 1:
                d[keymap[c[0]]] = None
            elif len(c) == 2:
                d[keymap[c[0]]] = c[1]
            else:
                raise ValueError('cannot parse filename: ' + filename)
        return d

    logger.info('read input directory: ' + args.input_dir)
    summary = []
    for input_file in os.listdir(args.input_dir):
        d = parse_filename(input_file)
        with h5py.File(os.path.join(args.input_dir, input_file), 'r') as f:
            d['accuracy'] = f['metrics/accuracy'][()]
            d['roc_auc'] = f['metrics/roc_auc'][()]
            summary.append(d)
    summary = pd.DataFrame.from_records(summary)
    with open_file_or_stdout(args.output_file) as fout:
        summary.to_csv(fout, sep='\t', index=False)

def summarize_metrics_by_rna(args):
    import numpy as np
    import h5py
    import pandas as pd
    from tqdm import tqdm
    from ioutils import open_file_or_stdout

    def parse_filename(filename):
        d = {}
        keymap = {'d': 'dataset', 'w': 'window_size', 'b': 'binarization_method', 'm': 'model', 'i': 'cv_index'}
        for v in filename.split(','):
            c = v.split('=')
            if len(c) == 1:
                d[keymap[c[0]]] = None
            elif len(c) == 2:
                d[keymap[c[0]]] = c[1]
            else:
                raise ValueError('cannot parse filename: ' + filename)
        return d

    logger.info('read input directory: ' + args.input_dir)
    summary = []
    for input_file in os.listdir(args.input_dir):
        d = parse_filename(input_file)
        metrics = pd.read_table(os.path.join(args.input_dir, input_file))
        for key, value in d.items():
            metrics[key] = value
        summary.append(metrics)
    summary = pd.concat(summary, axis=0)
    with open_file_or_stdout(args.output_file) as fout:
        summary.to_csv(fout, sep='\t', index=False)

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Train and test models')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('create_dataset')
    parser.add_argument('--input-file', '-i', type=str, required=True,
        help='input training data in HDF5 format. Each sequence is a dataset')
    parser.add_argument('--sequence-file', type=str, required=True, help='FASTA file')
    parser.add_argument('--cv-split-file', type=str)
    parser.add_argument('--output-file', '-o', type=str, required=True, help='output file in HDF5 format')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--window-size', type=int, default=50)
    parser.add_argument('--cutoff1', type=float)
    parser.add_argument('--cutoff2', type=float)
    parser.add_argument('--percentile', type=float, default=5)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--train-test-split', type=float, default=2,
        help='split the dataset into a training set and test based on the ratio if different from 2')
    parser.add_argument('--offset', type=int,
        help='the offset in the sequence to predict. Default is the middle point.')
    parser.add_argument('--seed', type=int,
        help='set the random seed for numpy')
    parser.add_argument('--dense-output', action='store_true',
        help='predict values for every position in the sequence')
    parser.add_argument('--min-coverage', type=float, default=0.5,
        help='minimum proportion of values that is not NaN for each sequence for dense output')

    parser = subparsers.add_parser('cv_split')
    parser.add_argument('--input-file', '-i', type=str, required=True,
        help='input training data in HDF5 format. Each sequence is a dataset')
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('--random-state', type=int)
    parser.add_argument('--output-file', '-o', type=str, required=True,
        help='output file in HDF5 format')

    parser = subparsers.add_parser('binarize')
    parser.add_argument('--input-file', '-i', type=str, required=True,
        help='input training data in HDF5 format. Each sequence is a dataset')
    parser.add_argument('--method', '-m', type=str, default='percentile:10,90')
    parser.add_argument('--output-file', '-o', type=str, required=True,
        help='output file in HDF5 format')

    parser = subparsers.add_parser('train')
    parser.add_argument('--input-file', '-i', type=str, required=True, help='input training data')
    parser.add_argument('--model-file', '-o', type=str, required=True, help='file path for saving the model')
    parser.add_argument('--model', '-m', type=str, required=True, help='model name')
    parser.add_argument('--regression', action='store_true', help='use regression model')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--xname', type=str, default='X_train', help='training data matrix name')
    parser.add_argument('--yname', type=str, default='y_train', help='training target name')
    parser.add_argument('--tensorboard-log-dir', type=str, help='output directory for TensorBoard')
    parser.add_argument('--keras-verbose', type=int, default=1, help='verbosity in the keras model.fit() function')
    parser.add_argument('--keras-log', type=str, help='CSV log file')
    parser.add_argument('--valid-file', type=str, help='validation data')
    parser.add_argument('--valid-xname', type=str, default='X_valid')
    parser.add_argument('--valid-yname', type=str, default='y_valid')

    parser = subparsers.add_parser('evaluate')
    parser.add_argument('--input-file', '-i', type=str, required=True, help='input training data')
    parser.add_argument('--model-file', type=str, required=True, help='file path for saving the model')
    parser.add_argument('--model-format', type=str, default='keras', choices=('keras', 'sklearn'))
    parser.add_argument('--output-file', '-o', type=str, help='output file to write the prediction and metrics to')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--swap-labels', action='store_true', help='swap 0/1 predictions')
    parser.add_argument('--metrics', type=list, default=['accuracy', 'roc_auc'],
        help='a metric name defined in sklearn.metrics.get_scorer')
    parser.add_argument('--cutoff', type=float, default=0.5, 
        help='cutoff for converting predictions to binary labels')
    parser.add_argument('--xname', type=str, default='X_test', 
        help='dataset name of the test data matrix')
    parser.add_argument('--yname', type=str, default='y_test',
        help='dataset name of the test labels')
    
    parser = subparsers.add_parser('evaluate_structures')
    parser.add_argument('--dot-file', type=str, required=True, help='structures in dot format')
    parser.add_argument('--pred-file', type=str, required=True, help='predictions from the predict command')
    parser.add_argument('--output-file', '-o', type=str, required=True)
    parser.add_argument('--swap-labels', action='store_true', help='swap structure labels before evaluation')

    parser = subparsers.add_parser('summarize_metrics')
    parser.add_argument('--input-dir', '-i', type=str, required=True,
        help='input directory containing all metrics')
    parser.add_argument('--output-file', '-o', type=str, default='-',
        help='a text summary table file')
    
    parser = subparsers.add_parser('summarize_metrics_by_rna')
    parser.add_argument('--input-dir', '-i', type=str, required=True,
        help='input directory containing all metrics')
    parser.add_argument('--output-file', '-o', type=str, default='-',
        help='a text summary table file')

    parser = subparsers.add_parser('predict')
    parser.add_argument('--input-file', '-i', type=str, required=True, help='sequence file in FASTA format'),
    parser.add_argument('--format', type=str, default='fasta',
        choices=('ct', 'ct_dir', 'fasta', 'rnafold', 'genomic_data'),
        help='input file format (fasta or ct)'),
    parser.add_argument('--model-file', type=str, required=True, help='file path for saving the model'),
    parser.add_argument('--offset', type=int, required=False, help='offset of the base in the window to predict'),
    parser.add_argument('--output-file', '-o', type=str, help='output file to write the prediction to'),
    parser.add_argument('--cutoff', type=float, default=0.5, help='cutoff for converting predictions to binary labels'),
    parser.add_argument('--batch-size', type=int, default=200),
    parser.add_argument('--dense', action='store_true'),
    parser.add_argument('--metrics', type=list, default='accuracy,roc_auc,ppv,sensitivity',
        help='a metric name defined in sklearn.metrics.get_scorer'),
    parser.add_argument('--fillna', type=float, help='fill missing values with a constant'),
    parser.add_argument('--alphabet', default='ATCG'),
    parser.add_argument('--swap-labels', action='store_true', help='swap 0/1 predictions'),
    parser.add_argument('--split', action='store_true', help='output separate files in the output directory'),
    parser.add_argument('--restraint-file', type=str, help='restraint file for RME'),
    parser.add_argument('--metric-file', type=str, help='prediction scores and metrics'),
    parser.add_argument('--metric-by-sequence-file', type=str, help='a text table with metrics calculated for each sequence'),
    parser.add_argument('--dense-pred-file', type=str, help='output dense predictions')

    args = main_parser.parse_args()
    if not args.command:
        raise ValueError('command is empty')
    logger = logging.getLogger('predict_reactivity.' + args.command)

    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'predict':
        predict(args)
    elif args.command == 'create_dataset':
        create_dataset(args)
    elif args.command == 'cv_split':
        cv_split(args)
    elif args.command == 'binarize':
        binarize(args)
    elif args.command == 'evaluate_structures':
        evaluate_structures(args)
    elif args.command == 'summarize_metrics':
        summarize_metrics(args)
    elif args.command == 'summarize_metrics_by_rna':
        summarize_metrics_by_rna(args)
