import numpy as np
import sys, os, argparse
import logging
import time
import h5py
from collections import namedtuple

from cmdtool import CommandLineTool, Argument
from ioutils import make_dir, append_extra_line
from formats import *
from genomic_data import GenomicData

def save_dataset(dataset, outfile):
    """Save a dataset to an HDF5 file
    dataset: a dict of numpy arrays
    """
    import h5py
    f = h5py.File(outfile, 'w')
    for key in dataset.keys():
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

def sequence_to_array(seq, alphabet='ATCG'):
    """output shape: (len(seq), 4)
    """
    sa = np.frombuffer(seq, dtype='S1')
    a = np.zeros((len(seq), 4), dtype='float32')
    for i, letter in enumerate(alphabet):
        a[:, i] = (sa == letter)

    return a

def array_to_sequence(a, alphabet='ATCG'):
    sa = np.zeros(a.shape[0], dtype='S1')
    for i, letter in enumerate(alphabet):
        sa[np.squeeze(a[:, i].astype('bool'))] = letter
    return str(np.getbuffer(sa))

def onehot_encode(x, alphabet):
    x_shape = list(x.shape)
    x = x.flatten()
    y = np.zeros((np.prod(x_shape), len(alphabet)), dtype='int32')
    for i, a in enumerate(alphabet):
        y[np.nonzero(x == a), i] = 1
    y = y.reshape(x_shape + [len(alphabet)])
    return y

def pwm_to_weblogo(logofile, pwm, alphabet):
    from cStringIO import StringIO
    f = StringIO()
    f.write('''NA motif0
XX
ID motif0
XX
''')
    f.write('P0 %s\n'%(' '.join(alphabet)))
    for i in range(pwm.shape[0]):
        f.write('P%d %s\n'%(i + 1, ' '.join(map(str, pwm[i, :]))))
    f.write('XX\n')
    f.write('//')

    weblogo_input = f.getvalue()

    import subprocess
    p = subprocess.Popen(['weblogo', '--format', 'png_print', '--datatype', 'transfac', '-o', logofile],
        stdin=subprocess.PIPE)
    p.communicate(weblogo_input)

def require_arguments(parsed_args, requires):
    parsed_args = vars(parsed_args)
    missing = []
    requires_args = [r.strip('--').replace('-', '_') for r in requires]
    args = {}
    for arg in requires_args:
        if parsed_args.get(arg) is None:
            missing.append(arg)
        else:
            args[arg] = parsed_args[arg]
    if len(missing) > 0:
        raise ValueError('arguments %s are required'%(', '.join(requires)))
    opt_args = {}
    for arg in parsed_args:
        if (arg not in requires_args) and (parsed_args.get(arg) is not None):
            opt_args[arg] = parsed_args[arg]
    return args, opt_args



class RollBackFile(object):
    def __init__(self, f, maxlines = 1):
        self.f = f
        self.maxlines = maxlines
        self.lines = [None for i in range(maxlines)]
        self.pos = self.maxlines

    def readline(self):
        if (self.pos >= self.maxlines) or (self.lines[self.pos] is None):
            line = self.f.readline()
            for i in range(self.maxlines - 1):
                self.lines[i] = self.lines[i + 1]
            self.lines[-1] = line
            self.pos = self.maxlines
        else:
            line = self.lines[self.pos]
            self.pos += 1
        return line

    def unreadline(self):
        if (self.pos == 0) or (self.lines[self.pos - 1] is None):
            raise ValueError('invalid ungetline')
        self.pos -= 1

    def close(self):
        self.f.close()

    def __next__(self):
        line = self.readline()
        if len(line) == 0:
            raise StopIteration()
        return line

    def __iter__(self):
        return self

class ProgressBar(object):
    def __init__(self, maxval=None, step=1, report_freq=1000, width=50, title='Progress',
            file=sys.stderr):
        self.maxval = maxval
        self.value = 0
        self.report_freq = report_freq
        self.curval = 0
        self.width = width
        self.title = title
        self.file = file
        self.start_time = time.clock()
        self.prev_time = self.start_time
        self._show = True

    def show(self, value=True):
        self._show = value

    def print_progress(self):
        if self.maxval is not None:
            pct = float(self.value)/self.maxval
        else:
            pct = 0
        self.file.write('\r\b{} ['.format(self.title))
        finished_width = int(pct*self.width)
        for i in range(finished_width):
            self.file.write('=')
        for i in range(self.width - finished_width):
            self.file.write(' ')
        cur_time = time.clock()
        elapsed = cur_time - self.start_time
        if self.maxval is not None:
            remaining = elapsed/self.value*(self.maxval - self.value)
            self.file.write('] {}/{}({:.2f}%), {:.1f}s elapsed, ETA {:.1f}s'.format(
                self.value, self.maxval, pct*100, elapsed, remaining))
        else:
            self.file.write('] {}, {:.1f}s elapsed'.format(
                self.value, elapsed))

    def update(self, value=1):
        self.value += value
        self.curval += value
        if self.curval >= self.report_freq:
            if self._show:
                self.print_progress()
            self.curval = 0

    def finish(self):
        if self._show:
            self.print_progress()
        self.file.write('\n')


def array_intersect(arrays):
    """Return the intersection (as a numpy array) of multiple numpy arrays.
    The elements in each array must be unique.
    """
    values, counts = np.unique(np.concatenate(arrays), return_counts=True)
    return values[counts == len(arrays)]

def array_union(arrays):
    """Return set union
    """
    return np.unique(np.concatenate(arrays))

def array_diff(a, b):
    """Return the set difference a - b
    """
    return np.unique(np.concatenate([a, b, b]))

def calc_dms_scores(treatment, control):
    """
    Reference: Ding, Y., Tang, Y., Kwok, C.K., Zhang, Y., Bevilacqua, P.C., and Assmann, S.M. (2014).
        In vivo genome-wide profiling of RNA secondary structure reveals novel regulatory features. Nature 505, 696-700.
    """
    treatment = np.log10(np.nan_to_num(treatment) + 1)
    control = np.log10(np.nan_to_num(control) + 1)
    P = treatment/((treatment.sum() + 1)/len(treatment))
    M = control/((control.sum() + 1)/len(control))
    scores = P - M
    scores[scores < 0] = 0
    pct2 = np.percentile(scores, 98)
    pct8 = np.percentile(scores, 92)
    scale = scores[np.logical_and(pct8 <= scores, scores <= pct2)].mean()
    if not np.isclose(scale, 0.0):
        scores /= scale
    scores[scores > 7.0] = 7.0
    return scores

def calc_pars_scores(v1, s1):
    """
    Reference: Wan, Y., Qu, K., Zhang, Q.C., Flynn, R.A., Manor, O., Ouyang, Z., Zhang, J., Spitale, R.C., Snyder, M.P., Segal, E., et al. (2014).
        Landscape and variation of RNA secondary structure across the human transcriptome. Nature 505, 706-709.
    """
    assert (v1.sum() != 0) and (s1.sum() != 0)
    v1_normed = np.log2(v1/v1.mean() + 5)
    s1_normed = np.log2(s1/s1.mean() + 5)
    scores = v1_normed - s1_normed
    return scores



def weighted_sum(sample_score, sample_weight, normalize=False):
    """Copied from https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/metrics/classification.py
    """
    if normalize:
        return np.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return np.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()

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
