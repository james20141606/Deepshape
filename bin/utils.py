import numpy as np
import h5py
from Bio import SeqIO
import subprocess
import itertools
import random

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

def fasta_to_onehot(filename, alphabet='AUCG', motif_only=False, parse_label=False):
    '''Read a FASTA file and convert the sequences to onehot encoding
    Args:
        motif_only: parse motif position from sequence name and extract only motif instances
        parse_label: parser the sequence names to determine whether the sequence is random
    Returns:
        ndarray of shape (n_sequences, seq_length, alphabet_size)
    '''

    onehot = Onehot(alphabet=alphabet)
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

def windows_to_sequences(X_split, starts, ends):
    '''Combine splitted windows into sequences
    Args:
        X_split, starts, ends: output of sequences_to_windows()
    Returns:
        a list of sequences
    '''
    X = []
    for i in range(len(starts)):
        X.append(X_split[starts[i]:ends[i]])
    return X


def pwm_to_transfac(counts, name, species_name='unknown', alphabet='ATCG'):
    '''Convert PWM to transfac format
    Args:
        counts: ndarray of shape (length, alphabet_size)
    Returns:
        svg script
    '''
    column_width =  max(int(np.ceil(np.log10(counts.shape[0]))),
                        int(np.ceil(np.log10(np.max(counts)))))
    body_format = '{:0>2d}\t{:<{width}d}\t{:<{width}d}\t{:<{width}d}\t{:<{width}d}'
    body_format = '{:0>2d}\t{:<{width}.1f}\t{:<{width}.1f}\t{:<{width}.1f}\t{:<{width}.1f}'
    header_format = '{:<{width}s}\t{:<{width}s}\t{:<{width}s}\t{:<{width}s}\t{:<{width}s}'
        
    lines = []
    lines.append('ID {}'.format(name))
    lines.append('XX')
    lines.append('BF {}'.format(species_name))
    lines.append('XX')
    lines.append(header_format.format('PO', *list(alphabet), width=column_width))
    for pos, counts_row in zip(range(1, counts.shape[0] + 1), counts):
        lines.append(body_format.format(pos, *counts_row, width=column_width))
    lines.append('XX')
    lines.append('//')
    transfac = '\n'.join(lines)
    return transfac

def pwm_to_weblogo(counts, name, species_name='unknown', alphabet='ATCG', output_format='png'):
    '''Convert PWM to WebLogo (SVG format)
    Args:
        counts: ndarray of shape (length, alphabet_size)
    Returns:
        svg script
    '''
    transfac = pwm_to_transfac(counts, name, species_name, alphabet)
    
    p = subprocess.Popen(['weblogo', '-D' 'transfac', '-F', output_format, '-s', 'large'],
                     stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE)
    weblogo_svg, _ = p.communicate(bytearray(transfac, encoding='UTF-8'))
    return weblogo_svg

def sample_pwm(pwm, alphabet='ATCG', size=1, return_onehot=False):
    '''Sample sequences from a PWM
    Args:
        pwm: probabilities. ndarray of shape [length, alphabet_size]
        size: number of sequences
    Returns:
        a list of sequences
    '''
    length, alphabet_size = pwm.shape
    assert len(alphabet) == alphabet_size
    alphabet = np.asarray(list(alphabet), dtype='U1')
    cumpwm = np.cumsum(pwm, axis=1)
    cumpwm[:, -1] += 0.1
    p = np.random.uniform(size=(size, length, alphabet_size))
    X = np.argmax(p < cumpwm[np.newaxis, :, :], axis=2)
    if return_onehot:
        X = (X[:, :, np.newaxis] == np.arange(alphabet_size)[np.newaxis, np.newaxis, :]).astype(np.int32)
    else:
        X = [''.join(a) for a in np.take(alphabet, X)]
            
    return X

def embed_pwm(pwm, alphabet='ATCG', size=1, return_onehot=False, length=None):
    '''Sample sequences from a PWM
    Args:
        pwm: probabilities. ndarray of shape [length, alphabet_size]
        size: number of sequences
        length: length of each sequence
    Returns:
        a list of sequences, start positions of each motifs
    '''
    motif_length = pwm.shape[0]
    if length is None:
        length = motif_length
    alphabet_size = pwm.shape[1]
    assert len(alphabet) == alphabet_size
    alphabet = np.asarray(list(alphabet), dtype='U1')
    cumpwm = np.cumsum(pwm, axis=1)
    cumpwm[:, -1] += 0.1
    p = np.random.uniform(size=(size, motif_length, alphabet_size))
    X_motif = np.argmax(p < cumpwm[np.newaxis, :, :], axis=2)
    if length > pwm.shape[0]:
        X = np.random.choice(alphabet_size, size=(size, length))
        starts = np.random.randint(length - motif_length, size=size)
        for i in range(size):
            X[i, starts[i]:(starts[i] + motif_length)] = X_motif[i]
    elif length == pwm.shape[0]:
        starts = np.zeros(size, dtype=np.int32)
        X = X_motif
    else:
        raise ValueError('cannot embed motif of length {} into sequence of length{}'.format(pwm.shape[0], length))
    if return_onehot:
        X = (X[:, :, np.newaxis] == np.arange(alphabet_size)[np.newaxis, np.newaxis, :]).astype(np.int32)
    else:
        X = [''.join(a) for a in np.take(alphabet, X)]
    return X, starts
    
def random_sequences(length, alphabet='ATCG', size=1, return_onehot=False):
    '''Generate random sequences of uniform nucleotide frequency
    Args:
        length: length of each generated sequence
        size: number of sequences to generate
    Returns:
        a list of sequences if return_onehot is False
        one-hot encoded ndarray of shape [size, length, len(alphabet)]
    '''
    alphabet = np.asarray(list(alphabet), dtype='U1')
    alphabet_size = len(alphabet)
    X = np.random.choice(alphabet_size, size=(size, length))
    if return_onehot:
        X = (X[:, :, np.newaxis] == np.arange(alphabet_size)[np.newaxis, np.newaxis, :]).astype(np.int32)
    else:
        X = [''.join(a) for a in np.take(alphabet, X)]
    return X

def set_keras_num_threads(n_threads):
    from keras import backend as K
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = n_threads
    config.inter_op_parallelism_threads = n_threads
    K.set_session(tf.Session(config=config))

'''Dinucleotide-preserving shuffle
Adapted from DeepBind: DeepBind-source/code/deepbind_util.py
'''
# form the graph from sequence
def form_seq_graph(seq):
    graph = {}
    for i, s in enumerate(seq[:-1]):
        if s not in graph:
            graph[s] = []
        graph[s].append(seq[i+1])
    return graph

# sample a random last edge graph
def sample_le_graph(graph, last_nt):
    le_graph = {}
    for vx in graph:
        le_graph[vx] = []
        if vx not in last_nt:
            le_graph[vx].append(random.choice(graph[vx]))
    return le_graph

# check whether there exists an Eulerian walk
# from seq[0] to seq[-1] in the shuffled
# sequence
def check_le_graph(le_graph, last_nt):
    for vx in le_graph:
        if vx not in last_nt:
            if not find_path(le_graph, vx, last_nt):
                return False
    return True

# function from: http://www.python.org/doc/essays/graphs/
# check whether there is a path between two nodes in a
# graph
def find_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if not graph.has_key(start):
        return None
    for node in graph[start]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
            if newpath: return newpath
        return None

# generate a new seq graph based on the last edge graph
# while randomly permuting all other edges
def form_new_graph(graph, le_graph, last_nt):
    new_graph = {}
    for vx in graph:
        new_graph[vx] = []
        temp_edges = graph[vx]
        if vx not in last_nt:
            temp_edges.remove(le_graph[vx][0])
        random.shuffle(temp_edges)
        for ux in temp_edges:
            new_graph[vx].append(ux)
        if vx not in last_nt:
            new_graph[vx].append(le_graph[vx][0])
    return new_graph

# walk through the shuffled graph and make the
# new sequence
def form_shuffled_seq(new_graph, init_nt, len_seq):
    is_done = False
    new_seq = init_nt
    while not is_done:
        last_nt  = new_seq[-1]
        new_seq += new_graph[last_nt][0]
        new_graph[last_nt].pop(0)
        if len(new_seq) >= len_seq:
            is_done = True
    return new_seq

# verify the nucl
def verify_counts(seq, shuf_seq):
    kmers = {}
    # Forming the k-mer library
    kmer_range = range(1,3)
    for k in kmer_range:
        for tk in itertools.product('ACGTN', repeat=k):
            tkey = ''.join(i for i in tk)
            kmers[tkey] = [0,0]

    kmers[seq[0]][0] = 1
    kmers[shuf_seq[0]][1] = 1
    for k in kmer_range:
        for l in range(len(seq)-k+1):
            tkey = seq[l:l+k]
            kmers[tkey][0] += 1
            tkey = shuf_seq[l:l+k]
            kmers[tkey][1] += 1
    for tk in kmers:
        if kmers[tk][0] != kmers[tk][1]:
            return False
    return True

_preprocess_seq = ['N']*256
_preprocess_seq[ord('a')] = _preprocess_seq[ord('A')] = 'A'  # Map A => A
_preprocess_seq[ord('c')] = _preprocess_seq[ord('C')] = 'C'  # Map C => C
_preprocess_seq[ord('g')] = _preprocess_seq[ord('G')] = 'G'  # Map G => G
_preprocess_seq[ord('t')] = _preprocess_seq[ord('T')] = 'T'  # Map T => T
_preprocess_seq[ord('u')] = _preprocess_seq[ord('U')] = 'T'  # Map U => T
_preprocess_seq = "".join(_preprocess_seq)

def preprocess_seq(seq):
    '''Convert sequence to DNA alphabet
    '''
    return seq.translate(_preprocess_seq)

def doublet_shuffle(seq, verify=False):
    seq = preprocess_seq(seq)
    last_nt = seq[-1]
    graph = form_seq_graph(seq)
    # sample a random last edge graph
    is_ok = False
    while not is_ok:
        le_graph = sample_le_graph(graph, last_nt)
        # check the last edge graph
        is_ok = check_le_graph(le_graph, last_nt)
    new_graph = form_new_graph(graph, le_graph, last_nt)
    shuf_seq  = form_shuffled_seq(new_graph, seq[0], len(seq))
    if verify:
        assert(verify_counts(seq, shuf_seq))
    return shuf_seq