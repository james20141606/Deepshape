#! /usr/bin/env python
import sys, os, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
from itertools import izip
from common import read_fasta, require_arguments, read_hdf5, make_dir, Argument
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('analyze_icshape')
from cmdtool import CommandLineTool, Argument

class Analysis(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def from_argv(cls, analysis_name, argv):
        """Parse command line arguments and store the parsed values as attributes
        """
        c = [c for c in Analysis.__subclasses__() if c.__name__ == analysis_name]
        if len(c) == 0:
            raise ValueError('unknown analysis: {}'.format(analysis_name))
        c = c[0]

        parser = argparse.ArgumentParser(c.description)
        parser.add_argument('analysis_name', type=str)
        for arg in c.arguments:
            parser.add_argument(arg.long_opt, dest=arg.name,
                type=arg.type, required=arg.required,
                default=arg.default, help=arg.help)
        args = parser.parse_args(argv)
        parsed_args = {}
        for arg in c.arguments:
            value = getattr(args, arg.name)
            parsed_args[arg.name] = arg.get_value(value)
        else:
            return c(**parsed_args)

    def analyze(self):
        raise NotImplementedError()

class IcshapeBaseDistribution(CommandLineTool):
    positionals = [('icshape_file', str)]
    options = [('pct', int)]
    def __init__(self, icshape_file, seqfile, outfile,
            pct=5, **kwargs):
        self.icshape_file = icshape_file
        self.seqfile = seqfile
        self.alphabet = 'ATCG'
        self.outfile = outfile
        super(IcshapeBaseDistribution, self).__init__(**kwargs)

    def analyze(self):
        icshape_f = h5py.File(self.icshape_file, 'r')
        icshape = icshape_f['icshape'][:]
        start = icshape_f['start'][:]
        end = icshape_f['end'][:]
        name = icshape_f['name'][:]
        rpkm = icshape_f['rpkm'][:]
        values = []
        for i in range(len(name)):
            values.append(icshape[start[i]:end[i]])
        sequences = dict(read_fasta(self.seqfile))
        sequences = [np.frombuffer(sequences[a], dtype='S1') for a in name]
        icshape_valid = icshape[np.logical_not(np.isnan(icshape))]
        cutoff1 = np.percentile(icshape_valid, 5)
        cutoff2 = np.percentile(icshape_valid, 95)
        self.plot_base_dist(sequences, values, range(-7, 8), cutoff1=cutoff1, cutoff2=cutoff2)

    def get_base_dist(self, seqs, values, offset=0, cutoff1=0, cutoff2=1):
        base_dist = np.zeros((2, 4), dtype='int64')
        for seq, values_ts in izip(seqs, values):
            ind_valid = np.logical_not(np.isnan(values_ts))
            ind_one  = np.nonzero(ind_valid & (values_ts >= cutoff2))[0] + offset
            ind_one  = ind_one[(ind_one >= 0) & (ind_one < len(seq))]
            ind_zero = np.nonzero(ind_valid & (values_ts <= cutoff1))[0] + offset
            ind_zero = ind_zero[(ind_zero >= 0) & (ind_zero < len(seq))]
            for i in range(len(self.alphabet)):
                if len(ind_zero) > 0:
                    base_dist[0, i] += (seq[ind_zero] == self.alphabet[i]).sum()
                if len(ind_one) > 0:
                    base_dist[1, i] += (seq[ind_one] == self.alphabet[i]).sum()
        return base_dist

    def plot_base_dist(self, sequences, values, offsets, cutoff1=0, cutoff2=1):
        fig, axes = plt.subplots(nrows=2, ncols=len(offsets), figsize=(20, 4), sharey=True)
        fig.tight_layout()
        for i, offset in enumerate(offsets):
            base_dist = self.get_base_dist(sequences, values,
                offset=offset, cutoff1=cutoff1, cutoff2=cutoff2).astype('float')
            for label in (0, 1):
                logger.debug('plot_base_dist: {}, {}'.format(label, offset))
                base_dist[label, :] /= base_dist[label, :].sum()
                ax = axes[label, i]
                ax.bar(np.arange(base_dist.shape[1]), base_dist[label, :], color='k', edgecolor='none', align='center')
                ax.set_xticks(np.arange(len(self.alphabet)))
                ax.set_xticklabels(self.alphabet)
                ax.set_ylabel('Density')
                ax.set_title('({}, {})'.format(label, offset))
        self.logger.info('savefig: {}'.format(self.outfile))
        plt.savefig(self.outfile, dpi=150, bbox_inches='tight')

class LogRegWeights(Analysis):
    def __init__(self, weights_file, outfile, **kwargs):
        self.weights_file = weights_file
        self.outfile = outfile
        self.alphabet = 'ATCG'
        super(LogRegWeights, self).__init__(**kwargs)

    def analyze(self):
        model_weights = h5py.File(self.weights_file, 'r')['/model_weights/dense_1/dense_1_W:0'][:]
        window_size = model_weights.shape[0]/len(self.alphabet)
        model_weights = model_weights.reshape((window_size, 4))
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 9), sharey=True)
        for i in range(len(self.alphabet)):
            ax = axes[i]
            ax.bar(np.arange(window_size), model_weights[:, i], color='b', edgecolor='none')
            ax.set_xticks(np.arange(window_size, step=5))
            ax.set_xlim(0, window_size)
            ax.set_title('Logistic regression weights ({})'.format(self.alphabet[i]))
            ax.set_ylabel('Weight')
        self.logger.info('savefig: {}'.format(self.outfile))
        plt.tight_layout()
        plt.savefig(self.outfile, dpi=150, bbox_inches='tight')

class IcshapeRpkmDistribution(Analysis):
    def __init__(self, icshape_file, outfile, title='RPKM Distribution', **kwargs):
        """
        Arguments:
            icshape_file: HDF5 file converted from icSHAPE output file
            outfile: a plot file
        """
        self.icshape_file = icshape_file
        self.outfile = outfile
        self.title = title
        super(IcshapeRpkmDistribution, self).__init__(**kwargs)

    def analyze(self):
        icshape_f = h5py.File(self.icshape_file, 'r')
        rpkm = icshape_f['rpkm'][:]
        icshape = icshape_f['icshape'][:]
        icshape_f.close()

        fig, ax = plt.subplots()
        ax.hist(np.log10(rpkm), bins=100)
        ax.set_xlabel('log10(RPKM)')
        ax.set_ylabel('Count')
        ax.set_title(self.title)
        plt.tight_layout()
        self.logger.info('savefig: {}'.format(self.outfile))
        plt.savefig(self.outfile, dpi=150)

class IcshapeDistributionWithRpkm(Analysis):
    def __init__(self, icshape_file, outfile, n_pct=10, n_cols=5,
            title='icSHAPE distribution with RPKM', **kwargs):
        """
        Arguments:
            icshape_file: HDF5 file converted from icSHAPE output file
            outfile: a plot file
        """
        self.icshape_file = icshape_file
        self.outfile = outfile
        self.title = title
        self.n_pct = n_pct
        self.n_cols = n_cols
        super(IcshapeDistributionWithRpkm, self).__init__(**kwargs)

    def analyze(self):
        icshape_f = h5py.File(self.icshape_file, 'r')
        rpkm = icshape_f['rpkm'][:]
        icshape = icshape_f['icshape'][:]
        start = icshape_f['start'][:]
        end = icshape_f['end'][:]
        icshape_f.close()

        rpkm_sorted_index = np.argsort(rpkm)
        bin_size = len(rpkm)/self.n_pct
        if bin_size*self.n_pct < len(rpkm):
            bin_size += 1
        n_rows = self.n_pct/self.n_cols
        fig, axes = plt.subplots(nrows=n_rows, ncols=self.n_cols,
            figsize=(self.n_cols*2 + 2, n_rows*2 + 1), sharey=True, sharex=True)
        for i in range(self.n_pct):
            if i == self.n_pct - 1:
                index = rpkm_sorted_index[i*bin_size:]
            else:
                index = rpkm_sorted_index[i*bin_size:(i+1)*bin_size]
            values = []
            for j in index:
                values_ts = icshape[start[j]:end[j]]
                values_ts = values_ts[np.logical_not(np.isnan(values_ts))]
                values.append(values_ts)
            values = np.concatenate(values)
            axes[i/self.n_cols, i%self.n_cols].hist(values, normed=True, bins=20)
            axes[i/self.n_cols, i%self.n_cols].set_title('RPKM {}% ({} values)'.format(100.0*float(i + 1)/self.n_pct, len(values)))
            #axes[i/self.n_cols, i%self.n_cols].set_yticks([])
            axes[i/self.n_cols, i%self.n_cols].set_xlabel('icSHAPE score')
        plt.tight_layout()
        self.logger.info('savefig: {}'.format(self.outfile))
        plt.savefig(self.outfile, dpi=150)

class IcshapeTrackWithRtStop(Analysis):
    description = 'Compare icSHAPE values with background/target RT stop counts'
    arguments = [Argument('icshape_file', required=True, help='icSHAPE file in HDF5 format'),
        Argument('target_file', required=True, help='target RT stop file in HDF5 format'),
        Argument('background_file', required=True, help='target RT stop file in HDF5 format'),
        Argument('outfile', required=True, help='output plot file in PDF format'),
        Argument('names', type=list, help='comma-list sequence names'),
        Argument('max_plots', type=int, default=10, help='maximum number of plots to generate')]

    def __init__(self, **kwargs):
        super(IcshapeTrackWithRtStop, self).__init__(**kwargs)

    def get_values_by_name(self, obj, entry, name):
        index = np.nonzero(obj['name'] == name)[0][0]
        return obj[entry][obj['start'][index]:obj['end'][index]]

    def plot_tracks(self, name, window_size=200):
        tracks = {}
        tracks['icshape'] = self.get_values_by_name(self.icshape, 'icshape', name)
        tracks['icshape'][np.isnan(tracks['icshape'])] = -0.2
        tracks['background RT stop'] = self.get_values_by_name(self.background, 'rt_stop', name)[1:]
        tracks['background base density'] = self.get_values_by_name(self.background, 'base_density', name)[1:]
        tracks['target RT stop'] = self.get_values_by_name(self.target, 'rt_stop', name)[1:]
        tracks['target base density'] = self.get_values_by_name(self.target, 'base_density', name)[1:]
        fig, ax = plt.subplots(nrows=len(tracks), figsize=[12, len(tracks)*2], sharex=True)
        i = 0
        for track_name, track in tracks.iteritems():
            track_length = min(window_size, len(track))
            ax[i].set_title('{} ({})'.format(track_name, name))
            ax[i].bar(np.arange(track_length), track[:track_length], color='b', edgecolor='none')
            if track_name == 'icshape':
                ax[i].set_ylim(-0.2, 1)
            i += 1
        plt.tight_layout()
        return fig

    def analyze(self):
        self.icshape = read_hdf5(self.icshape_file)
        self.background = read_hdf5(self.background_file)
        self.target = read_hdf5(self.target_file)
        name_counts = np.unique(np.concatenate([self.icshape['name'], self.background['name'], self.target['name']]), return_counts=True)
        common_names = name_counts[0][name_counts[1] == 3]
        if self.names is None:
            self.names = common_names[:self.max_plots]
        from matplotlib.backends.backend_pdf import PdfPages
        self.logger.info('open pdf file: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        with PdfPages(self.outfile) as pdf:
            for seqname in self.names:
                self.logger.info('plot track: {}'.format(seqname))
                pdf.savefig(self.plot_tracks(seqname))

if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis_name', type=str, choices=analyses.keys())
    parser.add_argument('--icshape-file', type=str, required=False,
        help='icSHAPE file in HDF5 format')
    parser.add_argument('--seq-file', type=str, required=False,
        help='sequence file in FASTA format')
    parser.add_argument('--out-dir', type=str, required=False,
        help='output directory')
    parser.add_argument('--prefix', type=str, required=False,
        help='prefix for output file names')
    parser.add_argument('-o', '--outfile', type=str, required=False,
        help='generic output file name')
    parser.add_argument('-i', '--infile', type=str, required=False,
        help='generic input file name')
    parser.add_argument('-t', '--title', type=str, required=False,
        help='plot title')
    args = parser.parse_args()
    """
    analyses = dict((c.__name__, c) for c in Analysis.__subclasses__())
    if len(sys.argv) < 2:
        print >>sys.stderr, 'Usage: {} analysis_name [options]'.format(sys.argv[0])
        print >>sys.stderr, 'Avaiable analysis names: ' + ' '.join(analyses)
        sys.exit(1)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 8
    Analysis.from_argv(sys.argv[1], sys.argv[1:]).analyze()


    """
    if args.analysis_name == 'IcshapeBaseDistribution':
        args, optargs = require_arguments(args, ['--icshape-file', '--seq-file', '--outfile'])
        IcshapeBaseDistribution(args['icshape_file'], args['seq_file'], args['outfile']).analyze()
    elif args.analysis_name == 'LogRegWeights':
        args, optargs = require_arguments(args, ['--infile', '--outfile'])
        LogRegWeights(args['infile'], args['outfile']).analyze()
    elif args.analysis_name == 'RpkmDistribution':
        args, optargs = require_arguments(args, ['--infile', '--outfile'])
        RpkmDistribution(args['infile'], args['outfile'], **optargs).analyze()
    elif args.analysis_name == 'IcshapeDistributionWithRpkm':
        args, optargs = require_arguments(args, ['--infile', '--outfile'])
        IcshapeDistributionWithRpkm(args['infile'], args['outfile'], **optargs).analyze()
    elif args.analysis_name == 'IcshapeTrackWithRtStop':
        args, optargs = require_arguments(args, [''])
    """
