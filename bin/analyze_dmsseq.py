#! /usr/bin/env python
import sys, os, argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from common import read_fasta, read_hdf5, make_dir, \
    CommandLineTool, Argument, IndexedFastaReader, sequence_to_array, \
    calc_dms_scores, GenomicData, ProgressBar
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8

class DmsseqTrackWithRtstop(CommandLineTool):
    description = 'Plot DMS-seq RT stop counts'
    arguments = [Argument('treatment_file', type=str, required=True, help='HDF5 format'),
        Argument('control_file', type=str, required=True, help='HDF5 format'),
        Argument('sequence_file', type=str, required=True, help='FASTA format with index file'),
        Argument('outfile', required=True, help='output plot file in PDF format'),
        Argument('names', type=list, help='comma-list sequence names'),
        Argument('max_plots', type=int, default=1, help='maximum number of plots to generate'),
        Argument('window_size', type=int, default=160, help='maximum width of each track')]


    def get_values_by_name(self, obj, entry, name):
        index = np.nonzero(obj['name'] == name)[0][0]
        return obj[entry][obj['start'][index]:obj['end'][index]]

    def plot_value_track(self, ax, data, title='Values'):
        ax.set_title(title)
        track_width = min(self.window_size, len(data))
        ax.set_xlim(0, track_width)
        ax.bar(np.arange(track_width), data[:track_width], color='b', edgecolor='none')

    def plot_sequence_track(self, ax, name, title='Sequence'):
        if not hasattr(self, 'fasta_f'):
            self.fasta_f = IndexedFastaReader(self.sequence_file)
        seq = self.fasta_f.get(name)
        track_width = min(self.window_size, len(seq))
        seq = seq[:track_width]
        colormap = {'N': '#000000', 'A': '#0000ff', 'T': '#00ff00', 'C': '#ff0000', 'G': '#ffa500', 'U': '#00ff00'}
        colors = map(colormap.get, seq)
        ax.set_title(title)
        ax.set_xlim(0, track_width)
        ax.set_yticks([])
        ax.bar(np.arange(track_width), np.ones(track_width), color=colors, edgecolor='w', linewidth=0.2)

    def plot_tracks(self, name):
        fig, ax = plt.subplots(4, 1, figsize=[12, 6], sharex=True,
            gridspec_kw=dict(height_ratios=[0.04, 0.32, 0.32, 0.32]))
        self.plot_sequence_track(ax[0], name, title='Sequence ({})'.format(name))
        treatment = self.get_values_by_name(self.treatment, 'dmsseq', name)
        control = self.get_values_by_name(self.control, 'dmsseq', name)
        dms_scores = calc_dms_scores(treatment, control)
        self.plot_value_track(ax[1], data=dms_scores,
            title='DMS-seq scores ({})'.format(name))
        self.plot_value_track(ax[2],
            data=treatment,
            title='Treatment RT stop ({})'.format(name))
        self.plot_value_track(ax[3],
            data=control,
            title='Control RT stop ({})'.format(name))
        #plt.tight_layout()
        return fig

    def __call__(self):
        self.treatment = read_hdf5(self.treatment_file)
        self.control = read_hdf5(self.control_file)
        name_counts = np.unique(np.concatenate([self.treatment['name'], self.control['name']]), return_counts=True)
        common_names = name_counts[0][name_counts[1] == 2]
        if self.names is None:
            self.names = common_names[:self.max_plots]
        from matplotlib.backends.backend_pdf import PdfPages
        self.logger.info('open pdf file: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        with PdfPages(self.outfile) as pdf:
            for seqname in self.names:
                self.logger.info('plot track: {}'.format(seqname))
                pdf.savefig(self.plot_tracks(seqname))

class ReadIndexedFasta(CommandLineTool):
    description = 'read fasta file with index'
    arguments = [Argument('fasta_file', short_opt='-i', type=str, required=True),
        Argument('names', type=list, required=True),
        Argument('fasta_index_file', type=str)]
    def __call__(self):
        f = IndexedFastaReader(self.fasta_file, self.fasta_index_file)
        for name in self.names:
            seq = f.get(name)
            if seq is not None:
                print name + '\t' + f.get(name)
        f.close()

class DmsseqBaseDistribution(CommandLineTool):
    description = 'plot the base disribution around each position with high or low DMS-seq signal'
    arguments = [Argument('dmsseq_file', type=str, required=True, help='DMS-seq scores in HDF5 format'),
        Argument('sequence_file', type=str, required=True, help='FASTA format with index file'),
        Argument('outfile', required=True, help='output plot file in PDF format'),
        Argument('percentile', type=float, default=5, help='percentile cutoff to filter DMS-seq scores'),
        Argument('max_offset', type=int, default=5),
        Argument('alphabet', type=str, default='ATCG')]

    def __call__(self):
        self.logger.info('load DMS-seq scores from: {}'.format(self.dmsseq_file))
        dmsseq = GenomicData(self.dmsseq_file, ['dmsseq'])
        scores = dmsseq['dmsseq']
        cutoff1 = np.percentile(scores, self.percentile)
        cutoff2 = np.percentile(scores, 100 - self.percentile)
        self.logger.info('DMS-seq score cutoffs: {}-{}'.format(cutoff1, cutoff2))
        discard = np.logical_and(cutoff1 < scores, scores < cutoff2)
        scores[(scores <= cutoff1) & np.logical_not(discard)] = 0
        scores[(scores >= cutoff2) & np.logical_not(discard)] = 1
        fasta_f = IndexedFastaReader(self.sequence_file)
        # calculate base distribution
        self.logger.info('calculate base distribution')
        self.offsets = range(-self.max_offset, self.max_offset + 1)
        base_dist = np.zeros([len(self.offsets), 2, 4], dtype='int64')
        progress = ProgressBar(len(dmsseq.names), title='')
        for name in dmsseq.names:
            seq = np.frombuffer(fasta_f[name], dtype='S1')
            values = dmsseq.feature('dmsseq', name)
            ind_valid = (np.logical_not(np.isnan(values)))[0]
            ind_one_ts = np.nonzero(values == 1)[0]
            ind_zero_ts = np.nonzero(values == 0)[0]
            for i_offset, offset in enumerate(self.offsets):
                ind_one  = ind_one_ts + offset
                ind_one  = ind_one[(ind_one >= 0) & (ind_one < len(seq))]
                ind_zero = ind_zero_ts + offset
                ind_zero = ind_zero[(ind_zero >= 0) & (ind_zero < len(seq))]
                for i in range(len(self.alphabet)):
                    if len(ind_zero) > 0:
                        base_dist[i_offset, 0, i] += (seq[ind_zero] == self.alphabet[i]).sum()
                    if len(ind_one) > 0:
                        base_dist[i_offset, 1, i] += (seq[ind_one] == self.alphabet[i]).sum()
            progress.update()
        progress.finish()
        fasta_f.close()
        base_dist = base_dist.astype('float64')
        # plot
        fig, axes = plt.subplots(nrows=2, ncols=len(self.offsets), figsize=(20, 4), sharey=True)
        fig.tight_layout()
        for i, offset in enumerate(self.offsets):
            for label in (0, 1):
                self.logger.debug('plot_base_dist: {}, {}'.format(label, offset))
                base_dist[i, label, :] /= base_dist[i, label, :].sum()
                ax = axes[label, i]
                ax.bar(np.arange(len(self.alphabet)), base_dist[i, label, :], color='k', edgecolor='none', align='center')
                ax.set_xticks(np.arange(len(self.alphabet)))
                ax.set_xticklabels(self.alphabet)
                ax.set_ylabel('Density')
                ax.set_title('({}, {})'.format(label, offset))
        self.logger.info('savefig: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        plt.savefig(self.outfile, dpi=150, bbox_inches='tight')

if __name__ == '__main__':
    commands = dict((c.__name__, c) for c in CommandLineTool.__subclasses__())
    if len(sys.argv) < 2:
        print >>sys.stderr, 'Usage: {} command [options]'.format(sys.argv[0])
        print >>sys.stderr, 'Avaiable commands: ' + ' '.join(commands.keys())
        sys.exit(1)
    CommandLineTool.from_argv()()
