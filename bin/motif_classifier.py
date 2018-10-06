#! /usr/bin/env python
from __future__ import print_function
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')

def set_keras_num_threads(n_threads):
    from keras import backend as K
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = n_threads
    config.inter_op_parallelism_threads = n_threads
    K.set_session(tf.Session(config=config))

def read_transfac(filename):
    record = {}
    with open(filename, 'r') as f:
        pfm = {}
        for line in f:
            tag = line[:2]
            if tag in ('XX', '//'):
                continue
            content = line[3:].strip()
            if tag.isdigit():
                pfm[int(tag)] = [float(a) for a in content.split()]
            elif tag == 'PO':
                record[tag] = content.split()
            else:
                record[tag] = content
        pfm = [pfm[i] for i in range(1, len(pfm) + 1)]
        pfm = np.asarray(pfm)
        record['LEN'] = pfm.shape[0]
        record['PFM'] = pfm
        record['NSEQ'] = sum(pfm[0])
        p = (pfm + 1)/np.sum(pfm + 1, axis=1, keepdims=True)
        record['PWM'] = p
        record['INFOBIT'] = -np.sum(p*np.log(p))
    return record

class PwmMotif(object):
    def __init__(self, length=3, alphabet='ATCG', name='motif'):
        self.length = length
        self.alphabet = np.asarray(list(alphabet), dtype='U1')
        self.name = name
        self.generate()
    
    def generate(self, alpha=0.5):
        pwm = np.random.dirichlet([alpha]*len(self.alphabet), size=self.length).T
        self.set_pwm(pwm, self.alphabet)

    def set_pwm(self, pwm, alphabet='ATCG'):
        '''
        Args:
            pwm: matrix of shape (alphabet_size, motif_length)
        '''
        self.alphabet = np.asarray(list(alphabet), dtype='U1')
        self.pwm = pwm
        self.length = pwm.shape[1]
        self.cumpwm = np.cumsum(self.pwm, axis=0)
        self.cumpwm[-1, :] += 0.1
        self.consensus = ''.join(self.alphabet[np.argmax(self.pwm, axis=0)])
        
    def __str__(self):
        lines = []
        lines.append('\t'.join(list(self.alphabet)))
        for i in range(self.length):
            lines.append('\t'.join(['%.6f'%a for a in self.pwm[:, i]]))
        lines.append('')
        return '\n'.join(lines)
    
    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as fin:
            pwm = []
            for lineno, line in enumerate(fin):
                c = line.strip().split('\t')
                if lineno == 0:
                    alphabet = ''.join(c)
                else:
                    pwm.append([float(a) for a in c])
            pwm = np.asarray(pwm).T
            motif = PwmMotif(pwm.shape[0], alphabet=alphabet)
            motif.set_pwm(pwm, alphabet)
        return motif

    @classmethod
    def load_transfac(cls, filename, pseudo_count=1):
        record = {}
        with open(filename, 'r') as f:
            pfm = {}
            for line in f:
                tag = line[:2]
                if tag in ('XX', '//'):
                    continue
                content = line[3:].strip()
                if tag.isdigit():
                    pfm[int(tag)] = [float(a) for a in content.split()]
                elif tag == 'PO':
                    record[tag] = content.split()
                else:
                    record[tag] = content
            pfm = [pfm[i] for i in range(1, len(pfm) + 1)]
            pfm = np.asarray(pfm)
            pwm = (pfm + pseudo_count)/np.sum(pfm + pseudo_count, axis=1, keepdims=True)
            pwm = pwm.T
            alphabet = ''.join(record['PO'])
            motif = PwmMotif(pwm.shape[0], alphabet=alphabet, name=record['ID'])
            motif.set_pwm(pwm, alphabet)
            return motif
            
    def sample(self, length, size=1, return_sequences=False):
        assert length >= self.length
        X = np.random.randint(len(self.alphabet), size=(size, length))
        p = np.random.uniform(size=(size, len(self.alphabet), self.length))
        motifs = np.argmax(p < self.cumpwm[np.newaxis, :, :], axis=1)
        positions = np.random.randint(length - self.length + 1, size=size)
        for i in range(size):
            X[i, positions[i]:(positions[i] + self.length)] = motifs[i]
        if return_sequences:
            X = [''.join(a) for a in np.take(self.alphabet, X)]
        else:
            X = (X[:, :, np.newaxis] == np.arange(len(self.alphabet))[np.newaxis, np.newaxis, :]).astype(np.int32)
        return X, positions

    def sample_negative(self, length, size=1):
        X = np.random.randint(len(self.alphabet), size=(size, length))
        X = (X[:, :, np.newaxis] == np.arange(len(self.alphabet))[np.newaxis, np.newaxis, :]).astype(np.float32)
        return X

class Onehot(object):
    def __init__(self, alphabet='AUCG'):
        self.alphabet = np.asarray(list(alphabet), dtype='U1')

    def encode(self, s):
        x = np.asarray(list(s), dtype='U1')
        encoded = (x[:, np.newaxis] == self.alphabet[np.newaxis, :]).astype(np.int32)
        return encoded

    def decode(self, x):
        return ''.join(self.alphabet[np.argmax(x, axis=1)])

def fasta_to_onehot(filename, motif_only=False, parse_label=False):
    '''Read a FASTA file and convert the sequences to onehot encoding
    Args:
        motif_only: parse motif position from sequence name and extract only motif instances
        parse_label: parser the sequence names to determine whether the sequence is random
    Returns:
        ndarray of shape (n_sequences, seq_length, alphabet_size)
    '''
    from Bio import SeqIO

    onehot = Onehot()
    dataset = []
    labels = []
    for record in SeqIO.parse(filename, 'fasta'):
        if motif_only:
            start, end = [int(a) for a in record.id.split('/')[1].split('-')]
            start -= 1
            dataset.append(onehot.encode(str(record.seq)[start:end])[np.newaxis, :, :])
        else:
            dataset.append(onehot.encode(str(record.seq))[np.newaxis, :, :])
        if parse_label:
            if record.id.startswith('random_') or record.id.startswith('RN_'):
                labels.append(0)
            else:
                labels.append(1)
    lengths = np.asarray([a.shape[1] for a in dataset])
    max_length = np.max(lengths)
    if not np.all(lengths == max_length):
        dataset_container = np.zeros((len(dataset), max_length, 4), dtype=np.int32)
        for i, a in enumerate(dataset):
            dataset_container[i, :a.shape[1], :] = a[0]
        dataset = dataset_container
    else:
        dataset = np.concatenate(dataset, axis=0)
    if parse_label:
        labels = np.asarray(labels, dtype=np.int32)
        return dataset, labels
    else:
        return dataset

def save_datasets(datasets, filename):
    with h5py.File(filename, 'w') as f:
        for cm_id, dataset in datasets.items():
            f.create_dataset(cm_id, data=dataset.astype(np.int8))

def fit_window(X):
    '''Fit X so that the window_size is exponential of 2
    '''
    window_size = X.shape[1]
    window_size2 = 2**int(np.ceil(np.log2(X.shape[1])))
    X2 = np.zeros((X.shape[0], window_size2, X.shape[2]), dtype=X.dtype)
    for i in range(X.shape[0]):
        X2[i, :window_size] = X[i]
    return X2, window_size2

def sequences_to_windows(X, window_size):
    lengths = np.asarray([(x.shape[0] - window_size + 1) for x in X])
    ends = np.cumsum(lengths)
    starts = ends - lengths
    n_windows = np.sum(lengths)
    X_split = np.empty((n_windows, window_size, X[0].shape[1]), dtype=X[0].dtype)
    i = 0
    for x in X:
        for j in range(x.shape[0] - window_size + 1):
            X_split[i] = x[j:(j + window_size)]
            i += 1
    return X_split, starts, ends

def train_vae(args):
    import h5py
    import os
    import numpy as np
    from models import MotifVariationalAutoencoder

    logger.info('load sequences: ' + args.input_file)
    X = fasta_to_onehot(args.input_file, motif_only=True)
    #X, window_size = fit_window(X)
    window_size = X.shape[1]
    logger.info('window size: {}'.format(window_size))

    logger.info('create the model')
    set_keras_num_threads(2)
    model = MotifVariationalAutoencoder(window_size=window_size, n_channels=4, latent_size=args.latent_size,
        name=args.name, n_layers=args.n_layers, n_sampler=args.n_sampler)
    model.model.summary()

    logger.info('train the model')
    model.model.fit(X, X, batch_size=args.batch_size, epochs=args.epochs)
    logger.info('save the model: ' + args.model_file)
    model.model.save_weights(args.model_file)
    metrics = model.model.evaluate(X, X, batch_size=args.batch_size)
    logger.info('save metrics: ' + args.metric_file)
    with open(args.metric_file, 'w') as fout:
        for metric_name, metric in zip(model.model.metrics_names, metrics):
            logger.info('{}: {}'.format(metric_name, metric))
            fout.write('{}\t{}\n'.format(metric_name, metric))

def evaluate_vae(args):
    import h5py
    import os
    import numpy as np
    from models import MotifVariationalAutoencoder
    from sklearn.metrics import roc_auc_score

    logger.info('read positive sequences: ' + args.pos_file)
    X_pos = fasta_to_onehot(args.pos_file, motif_only=True)
    window_size = X_pos.shape[1]
    logger.info('window size: {}'.format(window_size))
    logger.info('read negative sequences: ' + args.neg_file)
    X_neg = fasta_to_onehot(args.neg_file)
    X_neg = X_neg[:, :window_size, :]
    
    logger.info('create the model')
    set_keras_num_threads(2)
    model = MotifVariationalAutoencoder(window_size=window_size, n_channels=4, latent_size=args.latent_size,
        name=args.name, n_layers=args.n_layers, n_sampler=args.n_sampler)
    model.model.summary()
    logger.info('load model weights: ' + args.model_file)
    model.model.load_weights(args.model_file)

    logger.info('evaluate the model')
    logL_pos = model.logL(X_pos, n_sampler=args.n_sampler, batch_size=args.batch_size)
    logL_neg = model.logL(X_neg, n_sampler=args.n_sampler, batch_size=args.batch_size)
    metrics = {}
    metrics['roc_auc'] = roc_auc_score(np.concatenate([np.ones(X_pos.shape[0], dtype=np.int32), np.zeros(X_neg.shape[0], dtype=np.int32)]),
        np.concatenate([logL_pos, logL_neg]))
    logger.info('save metrics: ' + args.metric_file)
    with open(args.metric_file, 'w') as fout:
        for metric_name, metric in metrics.items():
            logger.info('{}: {}'.format(metric_name, metric))
            fout.write('{}\t{}\n'.format(metric_name, metric))

def mix_random_sequences(args):
    import numpy as np
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    def sample_mixture(n=10, evidence_function='linear', p=0.5):
        '''Sample evidence and labels from a mixture model
        Args:
            n: number of samples
            evidence_function: map evidence values to mixture coefficients
        Returns:
            a tuple (evidence, labels)
        '''
        if evidence_function == 'linear':
            y = np.random.uniform(0, 1, size=n)
        elif evidence_function == 'sigmoid':
            y = 1.0/(1.0 + np.exp(-5.0*np.random.uniform(-1, 1, size=n)))
        elif evidence_function == 'constant':
            y = np.full(n, p)
        else:
            raise ValueError('invalid evidence function: ' + evidence_function)
        z = (np.random.uniform(0, 1, size=n) < y).astype(np.int32)
        return y, z

    alphabet = np.asarray(list(args.alphabet), dtype='U1')
    # generate sequences with uniform probability
    logger.info('read input file: ' + args.input_file)
    sequences = list(SeqIO.parse(args.input_file, 'fasta'))
    # add a label to sequence name
    sequences = [SeqRecord(Seq(seq=record.seq, id=record.id + ' 1')) for record in sequences]
    logger.info('mix random sequences')
    
    evidence, labels = sample_mixture(len(sequences), evidence_function=args.evidence_function,
        p=int(len(sequences)*args.percentage/100.0))
    for i, label in enumerate(labels):
        record = sequences[i]
        if label == 0:
            if args.method == 'uniform':
                random_seq = ''.join(alphabet[np.random.choice(len(alphabet), size=len(record.seq))])
            elif args.method == 'shuffle':
                random_seq = np.asarray(list(str(record.seq)), dtype='U1')
                np.random.shuffle(random_seq)
                random_seq = ''.join(random_seq)
            else:
                raise ValueError('invalid method: {}'.format(args.method))
            sequences[i] = SeqRecord(Seq(random_seq), id='random_{:06d} 0'.format(i), description='')
        else:
            sequences[i] = SeqRecord(record.seq, id=record.id + ' 1', description='')
    logger.info('create output file: ' + args.output_file)
    with open(args.output_file, 'w') as fout:
        SeqIO.write(sequences, fout, 'fasta')

def train_mix(args):
    import h5py
    import os
    import numpy as np
    from models import MotifClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score

    logger.info('read positive sequences: ' + args.pos_file)
    X_pos, y_true_pos = fasta_to_onehot(args.pos_file, parse_label=True)
    y_given_pos = np.ones(X_pos.shape[0], dtype=np.int32)
    logger.info('read negative sequences: ' + args.neg_file)
    X_neg, y_true_neg = fasta_to_onehot(args.neg_file, parse_label=True)
    y_true_neg = np.zeros(X_neg.shape[0], dtype=np.int32)
    y_given_neg = np.zeros(X_neg.shape[0], dtype=np.int32)
    logger.info('number of random sequences in positive dataset: {}'.format(np.sum(y_true_pos == 0)))
    logger.info('number of random sequences in negative dataset: {}'.format(np.sum(y_true_neg == 0)))

    # divide the sequence into training (first part) and test (second part)
    train_indices = np.concatenate([np.arange(int(X_pos.shape[0]*args.train_ratio)),
        np.arange(int(X_neg.shape[0]*args.train_ratio)) + X_pos.shape[0]])
    test_indices = np.setdiff1d(np.arange(X_pos.shape[0] + X_neg.shape[0]), train_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    X = np.concatenate([X_pos, X_neg], axis=0)
    y_given = np.concatenate([y_given_pos, y_given_neg], axis=0)
    y_true = np.concatenate([y_true_pos, y_true_neg], axis=0)
    X_train, y_train_given = X[train_indices], y_given[train_indices]
    X_test, y_test_given, y_test_true = X[test_indices], y_given[test_indices], y_true[test_indices]

    set_keras_num_threads(2)
    motif_classifier = MotifClassifier(window_size=512, n_classes=2, n_conv_layers=args.n_conv_layers)

    logger.info('train the model')
    try:
        motif_classifier.model.fit(X_train, y_train_given, epochs=args.epochs, batch_size=50)
    except KeyboardInterrupt:
        pass
    logger.info('save the model: ' + args.model_file)
    motif_classifier.model.save(args.model_file)
    logger.info('evaluate the model')
    y_pred = motif_classifier.model.predict(X_test, batch_size=50)
    logger.info('save metrics: ' + args.metrics_file)
    metrics = {
        'roc_auc_true': roc_auc_score(y_test_true, y_pred),
        'roc_auc_pos_true': roc_auc_score(y_test_true[y_test_given == 1], y_pred[y_test_given == 1]),
        'roc_auc_given': roc_auc_score(y_test_given, y_pred)
    }
    with open(args.metrics_file, 'w') as f:
        for metric_name, metric_value in metrics.items():
            line = '{}\t{}\n'.format(metric_name, metric_value)
            f.write(line)
            logger.info(line)

def train(args):
    import h5py
    import os
    import numpy as np
    from models import MotifClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score

    logger.info('read positive sequences: ' + args.pos_file)
    X_pos = fasta_to_onehot(args.pos_file)
    y_pos = np.ones(X_pos.shape[0], dtype=np.int32)
    logger.info('read negative sequences: ' + args.neg_file)
    X_neg = fasta_to_onehot(args.neg_file)
    y_neg = np.zeros(X_neg.shape[0], dtype=np.int32)

    '''
    if args.id_file is not None:
        logger.info('read CM id file: ' + args.id_file)
        with open(args.id_file, 'r') as f:
            cm_ids = f.read().split()
    else:
        cm_ids = [args.id]
    n_classes = len(cm_ids) + 1
    logger.info('number of classes: {}'.format(n_classes))
    logger.info('load sequences: ' + args.input_dir)
    datasets = {}
    for cm_id in ['random'] + cm_ids:
        dataset, = fasta_to_onehot(os.path.join(args.input_dir, cm_id + '.fa'))
        datasets[cm_id] = dataset
    '''
    # divide the sequence into training (first part) and test (second part)
    train_indices = np.concatenate([np.arange(int(X_pos.shape[0]*args.train_ratio)),
        np.arange(int(X_neg.shape[0]*args.train_ratio)) + X_pos.shape[0]])
    test_indices = np.setdiff1d(np.arange(X_pos.shape[0] + X_neg.shape[0]), train_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    '''
    X_train = []
    y_train = []
    y_train_true = []
    X_test = []
    y_test = []
    y_test_true = []
    cm_labels = {cm_id:i for i, cm_id in enumerate(datasets.keys())}
    for cm_id, dataset in datasets.items():
        n_train = int(dataset.shape[0]*args.train_ratio)
        if n_classes == 2:
            y = np.full(dataset.shape[0], cm_labels[cm_id], dtype=np.int32)
        elif n_classes > 2:
            y = np.zeros((dataset.shape[0], n_classes), dtype=np.int32)
            y[:, cm_labels[cm_id]] = 1
        X_train.append(dataset[:n_train])
        y_train.append(y[:n_train])
        X_test.append(dataset[n_train:])
        y_test.append(y[n_train:])
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    random_indices = np.random.permutation(X_train.shape[0])
    X_train = np.take(X_train, random_indices, axis=0)
    y_train = np.take(y_train, random_indices, axis=0)

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    random_indices = np.random.permutation(X_test.shape[0])
    X_test = np.take(X_test, random_indices, axis=0)
    y_test = np.take(y_test, random_indices, axis=0)
    '''
    set_keras_num_threads(2)
    motif_classifier = MotifClassifier(window_size=512, n_classes=2, n_conv_layers=args.n_conv_layers)
    #motif_classifier.model.summary()

    logger.info('train the model')
    try:
        motif_classifier.model.fit(X_train, y_train, epochs=args.epochs, batch_size=50)
    except KeyboardInterrupt:
        pass
    logger.info('save the model: ' + args.model_file)
    motif_classifier.model.save(args.model_file)
    logger.info('evaluate the model')
    y_pred = motif_classifier.model.predict(X_test, batch_size=50)
    logger.info('save metrics: ' + args.metrics_file)
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred)
    }
    with open(args.metrics_file, 'w') as f:
        for metric_name, metric_value in metrics.items():
            f.write('{}\t{}\n'.format(metric_name, metric_value))

def generate_pwm(args):
    from ioutils import open_file_or_stdout

    motif = PwmMotif(length=args.length, alphabet=args.alphabet)
    motif.generate(alpha=args.alpha)
    fout = open_file_or_stdout(args.output_file)
    fout.write(str(motif))

def sample_pwm(args):
    from ioutils import open_file_or_stdout

    logger.info('load PWM from file: ' + args.input_file)
    motif = PwmMotif.load_transfac(args.input_file)
    motif_name = motif.name
    sequences, positions = motif.sample(args.length, size=args.n_sequences, return_sequences=True)
    logger.info('create output file: ' + args.output_file)
    fout = open_file_or_stdout(args.output_file)
    seq_id = 1
    for sequence, position in zip(sequences, positions):
        fout.write('>{}_{}/{}-{}\n'.format(motif_name, seq_id, position + 1, position + motif.length))
        fout.write(sequence.replace('T', 'U'))
        fout.write('\n')
        seq_id += 1

def train_mm(args):
    import numpy as np
    from Bio import SeqIO
    from ioutils import open_file_or_stdin, open_file_or_stdout

    logger.info('read input file: ' + args.input_file)
    fin = open_file_or_stdin(args.input_file)
    alphabet = None
    # A: first letter frequency, B: transition frequency matrix
    A = None
    B = None
    K = None
    for record in SeqIO.parse(fin, 'fasta'):
        seq = np.asarray(list(str(record.seq)), dtype='U1')
        if alphabet is None:
            alphabet = np.unique(seq)
            K = alphabet.shape[0]
            A = np.zeros(K, dtype=np.int64)
            B = np.zeros((K, K), dtype=np.int64)
        seq_onehot = (seq[:, np.newaxis] == alphabet[np.newaxis, :]).astype(np.int32)
        dinuc_onehot = (seq_onehot[:-1, :, np.newaxis]*seq_onehot[1:, np.newaxis, :])
        A += seq_onehot[0]
        B += np.sum(dinuc_onehot, axis=0)
    # normalize
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    A /= np.sum(A)
    B /= np.sum(B, axis=1, keepdims=True)
    
    logger.info('create output file: ' + args.output_file)
    fout = open_file_or_stdout(args.output_file)
    fout.write('\t'.join(alphabet) + '\n')
    fout.write('\t'.join([str(a) for a in A]) + '\n')
    for i in range(K):
        fout.write('\t'.join([str(a) for a in B[i]]) + '\n')

def sample_mm(args):
    import numpy as np
    import numba
    from ioutils import open_file_or_stdout

    logger.info('read model file: ' + args.model_file)
    with open(args.model_file, 'r') as fin:
        B = []
        for lineno, line in enumerate(fin):
            if lineno == 0:
                alphabet = np.asarray(line.strip().split('\t'), dtype='U1')
                K = alphabet.shape[0]
            elif lineno == 1:
                A = np.asarray([float(a) for a in line.strip().split('\t')])
            elif lineno >= 2:
                B.append(np.asarray([float(a) for a in line.strip().split('\t')]))
        B = np.concatenate(B).reshape((K, K))
    A_cum = np.cumsum(A)
    B_cum = np.cumsum(B, axis=1)

    #@numba.jit
    def sample_sequence(length, A_cum, B_cum):
        '''Sample a sequences from a first-order Markov model
        Args:
            length: length of the sequence to sample
            A_cum: cumulative probabilities of the first letter
            B_cum: cumulative transition probabilities (along Axis 1)
        Returns:
            integer array of labels
        '''
        K = A_cum.shape[0]
        r = np.random.uniform(size=length)
        s = np.zeros(length, dtype=np.int32)
        for k in range(K):
            if r[0] <= A_cum[k]:
                s[0] = k
                break
        for i in range(1, length):
            s_prev = s[i - 1]
            for k in range(K):
                if r[i] <= B_cum[s_prev, k]:
                    s[i] = k
                    break
        return s

    logger.info('create output file: ' + args.output_file)
    fout = open_file_or_stdout(args.output_file)
    for i in range(args.n_sequences):
        fout.write('>' + args.format%i + '\n')
        s = ''.join(alphabet[sample_sequence(args.length, A_cum, B_cum)])
        fout.write(s)
        fout.write('\n')
    fout.close()

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='A classifier for motifs')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('train',
                                   help='train a classifier for sequences')
    parser.add_argument('--pos-file', type=str, required=True,
                        help='positive sequences in FASTA format')
    parser.add_argument('--neg-file', type=str, required=True,
                        help='negative sequences in FASTA format')
    parser.add_argument('--model-file', '-m', type=str, required=True,
                        help='keras model file')
    parser.add_argument('--metrics-file', '-o', type=str, required=True,
                        help='metrics file')
    parser.add_argument('--n-conv-layers', type=int, default=1,
                        help='number of convolutional layers')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='fraction of samples that forms training set')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of epochs')
    
    parser = subparsers.add_parser('train_mix',
                                   help='train a classifier for sequences with positives mixed with random sequences')
    parser.add_argument('--pos-file', type=str, required=True,
                        help='positive sequences in FASTA format')
    parser.add_argument('--neg-file', type=str, required=True,
                        help='negative sequences in FASTA format')
    parser.add_argument('--model-file', '-m', type=str, required=True,
                        help='keras model file')
    parser.add_argument('--metrics-file', '-o', type=str, required=True,
                        help='metrics file')
    parser.add_argument('--n-conv-layers', type=int, default=1,
                        help='number of convolutional layers')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='fraction of samples that forms training set')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of epochs')
    
    parser = subparsers.add_parser('train_vae',
                                   help='train a VAE model for sequences')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='input fasta file sampled by cmemit')
    parser.add_argument('--model-file', '-m', type=str, required=True,
                        help='keras model file')
    parser.add_argument('--metric-file', type=str, required=True,
                        help='metric file')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--latent-size', type=int, default=8)
    parser.add_argument('--n-sampler', type=int, default=10,
                        help='number of random latent variables to draw for each input sequence')
    parser.add_argument('--n-layers', type=int, default=1,
                        help='number of layers in the encoder/decoder')
    parser.add_argument('--name', type=str, default='conv', help='model name')

    parser = subparsers.add_parser('evaluate_vae',
                                   help='evaluate a VAE model on positive and negative sequences')
    parser.add_argument('--pos-file', type=str, required=True,
                        help='input FASTA file containing positive sequences')
    parser.add_argument('--neg-file', type=str, required=True,
                        help='input FASTA file containing negative sequences')
    parser.add_argument('--model-file', '-m', type=str, required=True,
                        help='keras model file')
    parser.add_argument('--metric-file', type=str, required=True,
                        help='metric file')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--latent-size', type=int, default=8)
    parser.add_argument('--n-sampler', type=int, default=10,
                        help='number of random latent variables to draw for each input sequence')
    parser.add_argument('--n-layers', type=int, default=1,
                        help='number of layers in the encoder/decoder')
    parser.add_argument('--name', type=str, default='conv', help='model name')

    parser = subparsers.add_parser('mix_random_sequences',
                                help='randomly replace sequences with random sequences')
    parser.add_argument('--input-file', '-i', type=str, required=True, help='input FASTA file')
    parser.add_argument('--method', '-m', type=str, choices=('uniform', 'shuffle'), default='uniform')
    parser.add_argument('--percentage', '-p', type=float, default=50,
        help='fraction of sequences to replace with random sequences')
    parser.add_argument('--alphabet', '-a', type=str, default='AUCG')
    parser.add_argument('--evidence-function', '-e', type=str, 
        choices=('linear', 'sigmoid', 'constant'))
    parser.add_argument('--output-file', '-o', type=str, required=True, help='output FASTA file')

    parser = subparsers.add_parser('sample_pwm', help='generate random sequences from a PWM')
    parser.add_argument('--input-file', '-i', type=str, required=True, help='PWM file')
    parser.add_argument('--n-sequences', '-n', type=int, default=10)
    parser.add_argument('--length', '-l', type=int, default=10)
    parser.add_argument('--output-file', '-o', type=str, default='-')

    parser = subparsers.add_parser('sample_transfac', help='generate random sequences from a PWM')
    parser.add_argument('--input-file', '-i', type=str, required=True, help='PWM in transfac format')
    parser.add_argument('--n-sequences', '-n', type=int, default=10)
    parser.add_argument('--length', '-l', type=int, default=10)
    parser.add_argument('--output-file', '-o', type=str, default='-')

    parser = subparsers.add_parser('generate_pwm', help='generate a random motif')
    parser.add_argument('--length', '-l', type=int, default=4)
    parser.add_argument('--alphabet', '-a', type=str, default='ATCG')
    parser.add_argument('--alpha', type=float, default=0.3, help='alpha parameter for dirichlet prior')
    parser.add_argument('--output-file', '-o', type=str, default='-')

    parser = subparsers.add_parser('train_mm',
        help='train a first-order Markov model on sequences')
    parser.add_argument('--input-file', '-i', type=str, default='-',
        help='input sequences in FASTA format')
    parser.add_argument('--output-file', '-o', type=str, default='-',
        help='model parameters')
    
    parser = subparsers.add_parser('sample_mm',
        help='sample sequences a first-order Markov model on sequences')
    parser.add_argument('--model-file', '-i', type=str, required=True,
        help='input sequences in FASTA format')
    parser.add_argument('--length', '-l', type=int, default=100,
        help='length of sequences to sample')
    parser.add_argument('--n-sequences', '-n', type=int, default=10,
        help='number of sequences to sample')
    parser.add_argument('--format', '-f', type=str, default='random_%06d',
        help='format for sequence names')
    parser.add_argument('--output-file', '-o', type=str, default='-',
        help='FASTA file')
    
    args = main_parser.parse_args()
    if args.command is None:
        raise ValueError('empty command')
    logger = logging.getLogger('motif_classifier.' + args.command)

    import numpy as np

    if args.command == 'train':
        train(args)
    elif args.command == 'train_mix':
        train_mix(args)
    elif args.command == 'train_vae':
        train_vae(args)
    elif args.command == 'evaluate_vae':
        evaluate_vae(args)
    elif args.command == 'mix_random_sequences':
        mix_random_sequences(args)
    elif args.command == 'sample_pwm':
        sample_pwm(args)
    elif args.command == 'generate_pwm':
        generate_pwm(args)
    elif args.command == 'train_mm':
        train_mm(args)
    elif args.command == 'sample_mm':
        sample_mm(args)
