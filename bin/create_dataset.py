#! /usr/bin/env python
import os, sys, argparse
from common import load_dataset, save_dataset, \
    array_to_sequence, sequence_to_array

def create_dataset_from_arrays(X, y, permutate=True, balanced=True):
    """Create a data set from list of samples
        X: a list of sequence arrays of shape (seq_length, 4)
        y: a list of class labels
            For dense output: each item is of shape (seq_length, 1)
        balanced: randomly pick samples from the class containing more samples.
            For dense output, randomly set the labels from the bigger class to nan.
    """
    n_samples = len(X)
    if n_samples == 0:
        return None
    seq_length = X[0].shape[0]
    X = np.concatenate(X)
    X = X.reshape((n_samples, seq_length, 4))
    y = np.asarray(y, dtype=np.float32)
    dense_output = True if len(y.shape) > 1 else False
    ind = np.arange(X.shape[0])
    if balanced:
        # dense output
        if dense_output:
            y_shape = y.shape
            y = y.flatten()
            ind_pos = np.nonzero(y == 1)[0]
            ind_neg = np.nonzero(y == 0)[0]
            if ind_pos.shape[0] > ind_neg.shape[0]:
                ind_nan = np.random.choice(ind_pos, size=(ind_pos.shape[0] - ind_neg.shape[0]), replace=False)
                y[ind_nan] = np.nan
            elif ind_pos.shape[0] < ind_neg.shape[0]:
                ind_nan = np.random.choice(ind_neg, size=(ind_neg.shape[0] - ind_pos.shape[0]), replace=False)
                y[ind_nan] = np.nan
            y = y.reshape(y_shape)
            print 'Positives/negatives: %d/%d'%((y == 1).sum(), (y == 0).sum())
        else:
            ind_pos = np.nonzero(y == 1)[0]
            ind_neg = np.nonzero(y == 0)[0]
            if ind_pos.shape[0] > ind_neg.shape[0]:
                np.random.shuffle(ind_pos)
                ind_pos = ind_pos[:ind_neg.shape[0]]
            elif ind_pos.shape[0] < ind_neg.shape[0]:
                np.random.shuffle(ind_neg)
                ind_neg = ind_neg[:ind_pos.shape[0]]
            ind = np.concatenate([ind_pos, ind_neg])
            print 'Positives(negatives): %d'%(ind_pos.shape[0])
    if permutate:
        np.random.shuffle(ind)
    X = X[ind, :]
    if len(y.shape) == 1:
        y = y[ind]
    else:
        y = y[ind, :]

    return (X, y)

def motif_detect(pos_seq_file, neg_seq_file):
    X = []
    y = []
    for label, seq_file in enumerate((pos_seq_file, neg_seq_file)):
        with open(seq_file, 'r') as f:
            for lineno, line in enumerate(f):
                if (lineno % 2) == 0:
                    name = line
                else:
                    X.append(sequence_to_array(line.strip(), alphabet))
                    y.append(label)
    return create_dataset_from_arrays(X, y)

def RNAfold(infile, window_size, predict_site=None, stride=1,
        dense_output=False, missing_rate=0.2):
    X = []
    y = []
    if not predict_site:
        predict_site = window_size/2 + 1
    with open(infile, 'r') as f:
        for lineno, line in enumerate(f):
            if (lineno % 3) == 0:
                name = line.strip()[1:]
            elif (lineno % 3) == 1:
                seq = line.strip()
            elif (lineno % 3) == 2:
                s = np.frombuffer(line.split()[0], dtype='S1')
                structure = (s != '.').astype(np.float32)
                seq = sequence_to_array(seq)
                for i in range(0, len(seq) - window_size + 1, stride):
                    X.append(seq[i:(i + window_size), :])
                    if dense_output:
                        values = np.expand_dims(structure[i:(i + window_size)], -1)
                        # randomly set a portion of the values to nan
                        if missing_rate > 1e-6:
                            values[np.random.binomial(1, missing_rate, size=values.shape).astype('bool')] = np.nan
                        y.append(values)
                    else:
                        y.append(structure[i + predict_site])
    return create_dataset_from_arrays(X, y, balanced=False)

def icSHAPE(infile, window_size,
        predict_site=None, stride=1, cutoffs=None, maxlines=0,
        dense_output=False, min_coverage=0.5):
    """Input file is a multi-part file contaning icSHAPE values and sequences.
    Each part begins with a line of two columns: (sequence name, length).
    Each body line contains three columns: (position, base, value).
    Missing values are indicated with NULL.
    Arguments:
        predict_site: position in the window to be used as target value.
            The middle point of the window will be used if set to None.
        dense_output: indicate whether to predict all output values for each position
        min_coverage: minimum fraction of missing values in a window in dense_output mode
    """
    if not predict_site:
        predict_site = int((window_size + 1)/2)
    X = []
    y = []
    with open(infile, 'r') as fin:
        seq = None
        for lineno, line in enumerate(fin):
            if (lineno > maxlines) and (maxlines != 0):
                break
            fields = line.strip().split()
            if len(fields) == 2:
                # convert the values of the last sequences into X, y
                if seq:
                    seq = str(seq)
                    if dense_output:
                        # split the fragment into overlapping windows
                        for i in range(0, len(seq) - window_size + 1, stride):
                            coverage = 1 - float(np.count_nonzero(np.isnan(values[i:(window_size + i)])))/window_size
                            if coverage  > min_coverage:
                                X.append(sequence_to_array(seq[i:(i + window_size)]))
                                y.append(np.expand_dims(values[i:(i + window_size)], -1))
                    else:
                        # split the sequence into overlapping windows
                        # discard windows with invalid values
                        for i in range(0, len(seq) - window_size + 1, stride):
                            if not np.isnan(values[i + predict_site]):
                                X.append(sequence_to_array(seq[i:(i + window_size)]))
                                y.append(values[i + predict_site])
                name = fields[0]
                length = int(fields[1])
                seq = bytearray(length)
                values = np.full(length, np.nan, dtype='float64')
            elif len(fields) == 3:
                pos = int(fields[0]) - 1
                seq[pos] = fields[1]
                if fields[2] == 'NULL':
                    continue
                values[pos] = float(fields[2])
                if values[pos] <= cutoffs[0]:
                    values[pos] = 1
                # values that fall in the middle of the range are not used
                elif values[pos] < cutoffs[1]:
                    values[pos] = np.nan
                else:
                    values[pos] = 0
    X, y = create_dataset_from_arrays(X, y, permutate=True)
    return X, y

if __name__ == '__main__':
    dataset_list = ('RNAfold', 'icSHAPE')
    parser = argparse.ArgumentParser(description='Create training and test datasets for Deepfold')
    parser.add_argument('dataset', type=str, choices=dataset_list)
    parser.add_argument('-i', '--infile', type=str, required=True,
        help='input file, depends on the command')
    parser.add_argument('--pos-file', type=str, required=False,
        help='input file for positive dataset')
    parser.add_argument('--neg-file', type=str, required=False,
        help='input file for negative dataset')
    parser.add_argument('-o', '--outfile', type=str, required=True,
        help='output file, depends on the command')
    parser.add_argument('--motif-length', type=int, required=False, default=5,
        help='the length of the filter in the first convolutional layer')
    parser.add_argument('--window-size', type=int, required=False, default=51,
        help='window size for create RNAfold dataset')
    parser.add_argument('--stride', type=int, required=False, default=1,
        help='distance between adjacent windows in the sequence scanning model')
    parser.add_argument('--dense-output', action='store_true', required=False, default=False,
        help='predict dense output (one value for every position)')
    parser.add_argument('--predict-site', type=int, required=False,
        help='position in the window to predict for RNAfold')
    parser.add_argument('--icshape-cutoff1', type=float, required=False, default=0.0,
        help='low cutoff for icSHAPE values that defines paired bases')
    parser.add_argument('--icshape-cutoff2', type=float, required=False, default=0.85,
        help='high cutoff for icSHAPE values that defines single-stranded bases')
    args = parser.parse_args()

    import numpy as np
    if args.dataset == 'icSHAPE':
        X, y = icSHAPE(args.infile, args.window_size,
            predict_site=args.predict_site,
            cutoffs=(args.icshape_cutoff1, args.icshape_cutoff2),
            stride=args.stride,
            dense_output=args.dense_output)
        save_dataset({'X': X, 'y': y}, args.outfile)
    elif args.dataset == 'RNAfold':
        X, y = RNAfold(args.infile, args.window_size,
            predict_site=args.predict_site,
            stride=args.stride,
            dense_output=args.dense_output)
        save_dataset({'X': X, 'y': y}, args.outfile)
