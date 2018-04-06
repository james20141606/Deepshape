#! /usr/bin/env python
from cmdtool import CommandLineTool, Argument
from ioutils import prepare_output_file
from formats import read_fasta
import sys, os

def import_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['legend.fontsize'] = 10
    from matplotlib.backends.backend_pdf import PdfPages
    globals().update(locals())

def read_background_rt(filename):
    import numpy as np
    rt_stop = {}
    base_density = {}
    length = {}
    base_frequency = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            c = line.strip().split('\t')
            length[c[0]] = int(c[1])
            base_frequency = map(float, c[3].split(','))
            if c[2] == 'RTstop':
                rt_stop[c[0]] = np.asarray(c[4:], dtype='float32')
            elif c[2] == 'baseDensity':
                base_density[c[0]] = np.asarray(c[4:], dtype='float32')
    return rt_stop, base_density, length, base_frequency

class BaseDensityCorrelationBetweenDatasets(CommandLineTool):
    description = 'Analyze the correlation of base density between icSHAPE datasets'
    arguments = [Argument('infile1', short_opt='-a', type=str, required=True,
            help='first rt file (e.g. background.normalized.rt)'),
        Argument('infile2', short_opt='-b', type=str, required=True,
            help='second rt file (e.g. background.normalized.rt)'),
        Argument('max_seq', type=int, default=10,
            help='maximum number of examples to plot'),
        Argument('max_len', type=int, default=400,
            help='maximum length to plot in the examples'),
        Argument('prefix', short_opt='-o', type=str, required=True)]
    def __call__(self):
        import numpy as np
        import h5py
        from scipy.stats import pearsonr
        import pandas as pd
        import itertools
        import_matplotlib()

        self.logger.info('read input file ' + self.infile1)
        _, base_density1, length1, _ = read_background_rt(self.infile1)
        self.logger.info('read input file ' + self.infile2)
        _, base_density2, length2, _ = read_background_rt(self.infile2)
        names = list(set(base_density1.iterkeys()) & set(base_density2.iterkeys()))
        names = np.asarray(names)
        r = []
        for name in names:
            r_, p_value = pearsonr(base_density1[name], base_density2[name])
            r.append(r_)
        r = np.asarray(r, dtype='float64')
        df = pd.DataFrame({'name': names, 'pearsonr': r})

        outfile = self.prefix + '.pearsonr.txt'
        self.logger.info('save Pearson correlation: ' + outfile)
        prepare_output_file(outfile)
        df.to_csv(outfile, sep='\t', index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(r, bins=50, edgecolor='none')
        ax.set_xlabel('Pearson correlation coefficient')
        ax.set_ylabel('Counts')
        ax.set_title('Correlation between base density of icSHAPE datasets')
        outfile = self.prefix + '.pearsonr.pdf'
        self.logger.info('save Pearson correlation plot: ' + outfile)
        plt.savefig(outfile)

        outfile = self.prefix + '.examples.pdf'
        self.logger.info('save examples plot: ' + outfile)
        selected_names = np.random.choice(names, size=self.max_seq, replace=True)
        from scipy import signal

        """
        with PdfPages(outfile) as pdf:
            for name in selected_names:
                self.logger.info('plot base density for ' + name)
                fig, ax = plt.subplots(2, 1, figsize=(12, 3), sharex=True)
                length = min(self.max_len, len(base_density1[name]))

                y = base_density1[name]
                ax[0].bar(np.arange(length), y[:length]/y.mean(), edgecolor='none')
                ax[0].set_xlabel('Position')
                ax[0].set_ylabel('Base density')
                ax[0].set_title(name)

                y = base_density2[name]
                ax[1].bar(np.arange(length), y[:length]/y.mean(), edgecolor='none')
                ax[1].set_xlabel('Position')
                ax[1].set_ylabel('Base density')
                ax[1].set_title(name)

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
                """

        with PdfPages(outfile) as pdf:
            for name in selected_names:
                self.logger.info('plot base density for ' + name)
                fig, axes = plt.subplots(2, 6, figsize=(15, 5), sharex=True)
                length = min(self.max_len, len(base_density1[name]))
                for i, j in itertools.product(range(2), range(6)):
                    if i == 0:
                        y = base_density1[name]
                    else:
                        y = base_density2[name]
                    std = j*20
                    if std > 0:
                        window = signal.gaussian(100, std=std)
                        y = signal.convolve(y, window, mode='same')
                    axes[i, j].bar(np.arange(length), y[:length]/y.mean(), edgecolor='none')
                    axes[i, j].set_xlabel('Position')
                    axes[i, j].set_ylabel('Base density')
                    axes[i, j].set_title('%s\n(std=%d)'%(name, std))
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

class CreateRegressDatasetForBaseDensity(CommandLineTool):
    description = 'Create a regression datasets from base density files'
    arguments = [Argument('infile', short_opt='-i', type=str, required=True,
            help='background rt file (e.g. background.normalized.rt)'),
        Argument('sequence_file', type=str, required=True,
            help='sequence in FASTA format'),
        Argument('max_samples', type=int, default=1000000,
            help='maximum number of samples'),
        Argument('test_ratio', type=float, default=0.2,
            help='ratio between the number of test samples'),
        Argument('window_size', type=int, default=100),
        Argument('offset', type=int),
        Argument('stride', type=int, default=1),
        Argument('smooth', action='store_true'),
        Argument('smooth_width', type=float, default=100),
        Argument('outfile', short_opt='-o', type=str, required=True)]

    def __call__(self):
        import numpy as np
        from sklearn.model_selection import train_test_split
        import h5py
        from common import sequence_to_array
        from scipy import signal

        self.logger.info('read input file: ' + self.infile)
        _, base_density, length, _ = read_background_rt(self.infile)
        names = base_density.keys()
        self.logger.info('read sequence file: ' + self.sequence_file)
        sequences = dict(read_fasta(self.sequence_file))

        if self.offset is None:
            self.offset = (self.window_size + 1)/2
        X = []
        y = []
        if self.smooth:
            self.logger.info('smooth the values using Gaussian window of width %.1f'%self.smooth_width)
            window = signal.gaussian(100, std=self.smooth_width)
        for name in names:
            seq = sequences[name]
            values = base_density[name]/base_density[name].mean()
            if self.smooth:
                # smooth the signal
                values = signal.convolve(values, window, mode='same')
            for i in range(0, len(seq) - self.window_size, self.stride):
                X.append(sequence_to_array(seq[i:(i + self.window_size)]))
                y.append(values[i + self.offset])
                if len(X) >= self.max_samples:
                    break
        n_samples = len(X)
        self.logger.info('created {} samples'.format(n_samples))

        X = np.concatenate(X)
        X = X.reshape((n_samples, self.window_size, 4))
        y = np.asarray(y, dtype='float32')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_ratio)

        self.logger.info('save file: ' + self.outfile)
        prepare_output_file(self.outfile)
        f = h5py.File(self.outfile, 'w')
        f.create_dataset('offset', data=int(self.offset))
        f.create_dataset('window_size', data=int(self.window_size))
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('X_test',  data=X_test)
        f.create_dataset('y_test', data=y_test)
        f.close()

if __name__ == '__main__':
    CommandLineTool.from_argv()()
