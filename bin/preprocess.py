#! /usr/bin/env python
import os, sys
from common import ProgressBar, \
    read_bed12, make_dir, array_intersect, calc_dms_scores, GenomicData, \
    IndexedFastaReader, calc_pars_scores, read_fasta, read_rnafold, \
    read_ct, sequence_to_array, append_extra_line, onehot_encode
from ioutils import prepare_output_file
from cmdtool import CommandLineTool, Argument
import numpy as np
import logging
import itertools
import h5py

def import_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    plt.rcParams['legend.fontsize'] = 11
    from matplotlib.backends.backend_pdf import PdfPages
    globals().update(locals())

logging.basicConfig(level=logging.DEBUG)

class ExtractBigWig(CommandLineTool):
    description = 'Extract values from bigwig files'
    arguments = [Argument('plus_bigwig_file', type=str, required=True),
        Argument('minus_bigwig_file', type=str),
        Argument('bed_file', type=str, required=True,
            help='transcript annotation in BED12 format'),
        Argument('outfile', short_opt='-o', type=str, required=True, help='HDF5 file'),
        Argument('min_coverage', type=float, default=0.0,
            help='minimum coverage to keep the transcript'),
        Argument('jkweb_path', type=str, default='lib/libjkweb.so'),
        Argument('feature_name', type=str, default='data',
            help='dataset name to be set in the output file')]
    def __call__(self):
        import bigwig
        import h5py

        bigwig.init_jkweb(self.jkweb_path)
        bw = {}
        self.logger.info('open file: {}'.format(self.plus_bigwig_file))
        bw['+'] = bigwig.BigWigFile(self.plus_bigwig_file)
        if self.minus_bigwig_file:
            self.logger.info('open file: {}'.format(self.minus_bigwig_file))
            bw['-'] = bigwig.BigWigFile(self.minus_bigwig_file)
        names = []
        values = []
        progress = ProgressBar(report_freq=100)
        for record in read_bed12(self.bed_file):
            values_ts = bw[record.strand].interval_query_blocked(record.chrom, record.chromStart, record.chromEnd,
                record.blockStarts, record.blockSizes)
            progress.update()
            if (self.min_coverage > 0) and (np.count_nonzero(np.isnan(values_ts)) > (1.0 - self.min_coverage)*len(values_ts)):
                continue
            if record.strand == '-':
                values_ts = values_ts[::-1]
            values.append(values_ts)
            names.append(record.name)
        progress.finish()
        for bwf in bw.itervalues():
            bwf.close()
        self.logger.info('number of sequences left after filtering: {}'.format(len(values)))

        length = np.asarray(map(len, values), dtype='int64')
        end = np.cumsum(length)
        self.logger.info('save file: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        GenomicData.from_data(np.asarray(names, dtype='S'), features={self.feature_name: values},
            create_namedict=False).save(self.outfile)


class CalcDmsseqScores(CommandLineTool):
    description = 'Calculate DMS-seq scores from RT stop counts of treatment and control samples'
    arguments = [Argument('treatment_file', type=str, required=True),
        Argument('control_file', type=str, required=True),
        Argument('outfile', short_opt='-o', type=str, required=True)]
    def __call__(self):
        self.logger.info('load treatment counts: {}'.format(self.treatment_file))
        treatment = GenomicData(self.treatment_file, ['dmsseq'])
        self.logger.info('load control counts: {}'.format(self.control_file))
        control = GenomicData(self.control_file, ['dmsseq'])
        names = array_intersect([treatment.names, control.names])
        scores = []
        mean_counts_treatment = np.zeros(len(names))
        mean_counts_control = np.zeros(len(names))
        for i, name in enumerate(names):
            scores.append(calc_dms_scores(treatment.feature('dmsseq', name),
                control.feature('dmsseq', name)))
            mean_counts_treatment[i] = np.nan_to_num(treatment.feature('dmsseq', name)).mean()
            mean_counts_control[i] = np.nan_to_num(control.feature('dmsseq', name)).mean()
        self.logger.info('save DMS-seq scores: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        GenomicData.from_data(names,
                              features={'dmsseq': scores},
                              meta={'mean_counts_treatment': mean_counts_treatment,
                                    'mean_counts_control': mean_counts_control},
            create_namedict=False).save(self.outfile)

class CalcParsScores(CommandLineTool):
    description = 'Calculate PARS scores from RT stop counts of V1 and S1 stop counts'
    arguments = [Argument('v1_file', type=str, required=True),
        Argument('s1_file', type=str, required=True),
        Argument('outfile', short_opt='-o', type=str, required=True),
        Argument('min_coverge', type=float, default=0.5)]
    def __call__(self):
        self.logger.info('load V1 stop counts: {}'.format(self.v1_file))
        v1 = GenomicData(self.v1_file, ['rt_stop'])
        self.logger.info('load S1 stop counts: {}'.format(self.s1_file))
        s1 = GenomicData(self.s1_file, ['rt_stop'])
        common_names = array_intersect([v1.names, s1.names])
        names = []
        scores = []
        for name in common_names:
            v1_rt = v1.feature('rt_stop', name)
            s1_rt = s1.feature('rt_stop', name)
            v1_rt[np.isnan(v1_rt)] = 0
            s1_rt[np.isnan(s1_rt)] = 0
            if (np.count_nonzero(v1_rt > 0) < self.min_coverge*len(v1_rt)) or \
                (np.count_nonzero(s1_rt > 0) < self.min_coverge*len(s1_rt)):
                continue
            scores.append(calc_pars_scores(v1_rt, s1_rt))
            names.append(name)
        self.logger.info('save PARS scores: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        GenomicData.from_data(names, features={'pars': scores},
            create_namedict=False).save(self.outfile)

class PlotTracks(CommandLineTool):
    description = 'Plot tracks'
    arguments = [Argument('track_file', type=str, action='append', help='HDF5 format'),
        Argument('feature', type=str, action='append'),
        Argument('title', type=str, action='append'),
        Argument('sequence_file', type=str, required=True, help='FASTA format with index file'),
        Argument('outfile', required=True, help='output plot file in PDF format'),
        Argument('names', type=list, help='comma-list sequence names'),
        Argument('max_plots', type=int, default=10, help='maximum number of plots to generate'),
        Argument('window_size', type=int, default=160, help='maximum width of each track')]

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
        seq = np.frombuffer(seq, dtype='S1')
        for base in 'ATGC':
            x = np.nonzero(seq == base)[0]
            ax.bar(x, np.ones(len(x)), color=colormap[base], edgecolor='w', linewidth=0.2, label=base)
        ax.set_xticks([])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.8), fontsize=8, ncol=4, borderpad=0, borderaxespad=0, frameon=False)
        #ax.bar(np.arange(track_width), np.ones(track_width), color=colors, edgecolor='w', linewidth=0.2)

    def plot_tracks(self, name):
        fig, ax = plt.subplots(self.n_tracks + 1, 1, figsize=[12, self.n_tracks*2 + 1], sharex=True,
            gridspec_kw=dict(height_ratios=[0.1] + [0.9/self.n_tracks] * self.n_tracks))
        with plt.rc_context(rc={'font.size': 8}):
            self.plot_sequence_track(ax[0], name, title='Sequence ({})'.format(name))
            for i in range(self.n_tracks):
                data = self.track_data[i].feature(self.feature[i], name)
                self.plot_value_track(ax[i + 1], data=data,
                    title='{} ({})'.format(self.title[i], name))
        plt.tight_layout()
        return fig

    def __call__(self):
        import_matplotlib()

        self.n_tracks = len(self.track_file)
        self.track_data = []
        for i in range(self.n_tracks):
            self.track_data.append(GenomicData(self.track_file[i], [self.feature[i]]))
        names = array_intersect([d.names for d in self.track_data])
        if self.names is None:
            self.names = names[:self.max_plots]
        from matplotlib.backends.backend_pdf import PdfPages
        self.logger.info('open pdf file: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        with PdfPages(self.outfile) as pdf:
            for seqname in self.names:
                self.logger.info('plot track: {}'.format(seqname))
                pdf.savefig(self.plot_tracks(seqname))

class BaseDistribution(CommandLineTool):
    description = 'plot the base disribution around each position with high or low values'
    arguments = [Argument('score_file', short_opt='-i', type=str, required=True, help='Scores in HDF5 format'),
        Argument('feature', type=str, required=True, help='feature name to use in the score file'),
        Argument('sequence_file', type=str, required=True, help='FASTA format with index file'),
        Argument('outfile', short_opt='-o', type=str, required=True, help='output plot file in PDF format'),
        Argument('percentile', type=float, default=5, help='percentile cutoff to filter DMS-seq scores'),
        Argument('cutoff1', type=float, help='the lower cutoff'),
        Argument('cutoff2', type=float, help='the upper cutoff'),
        Argument('max_offset', type=int, default=5),
        Argument('plot_type', type=str, default='stacked_bar', choices=('stacked_bar', 'separate_bar')),
        Argument('alphabet', type=str, default='ATCG')]

    def stacked_bar(self, base_dist):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        colors = ['#b30000', '#0000b3', '#b300b3', '#00b3b3']
        for label in (0, 1):
            base_fraction = np.zeros((len(self.alphabet), base_dist.shape[0]))
            for i_offset in range(len(self.offsets)):
                base_fraction[:, i_offset] = base_dist[i_offset, label, :]/base_dist[i_offset, label, :].sum()
            base_fraction_bottom = np.cumsum(base_fraction, axis=0)
            ax = axes[label]
            for base in range(len(self.alphabet) - 1, -1, -1):
                if base > 0:
                    ax.bar(np.arange(len(self.offsets)), base_fraction[base, :],
                        bottom=base_fraction_bottom[base - 1, :], color=colors[base], edgecolor='none',
                        align='center', label=self.alphabet[base])
                else:
                    ax.bar(np.arange(len(self.offsets)), base_fraction[base, :],
                        color=colors[base], edgecolor='none',
                        align='center', label=self.alphabet[base])
            ax.legend()
            ax.set_xlabel('Relative position')
            ax.set_xticks(np.arange(len(self.offsets)))
            ax.set_xticklabels(self.offsets)
            ax.set_ylabel('Base fraction')
            ax.set_xlim(-0.5, len(self.offsets) + 2)
            if label == 0:
                ax.set_title('Base distribution for low reactivity')
            else:
                ax.set_title('Base distribution for high reactivity')
        plt.tight_layout()
        self.logger.info('savefig: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        plt.savefig(self.outfile, dpi=150, bbox_inches='tight')

    def separate_bar(self, base_dist):
        fig, axes = plt.subplots(nrows=2, ncols=len(self.offsets), figsize=(20, 4), sharey=True)
        fig.tight_layout()
        for i, offset in enumerate(self.offsets):
            for label in (0, 1):
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

    def __call__(self):
        import_matplotlib()
        self.logger.info('load scores from: {}'.format(self.score_file))
        data = GenomicData(self.score_file, [self.feature])
        scores = data[self.feature]
        # if either cutoff1 or cutoff2 is not given,
        # then calculate the cutoffs from data and the given percentile
        if (self.cutoff1 is None) or (self.cutoff2 is None):
            scores_valid = scores[np.logical_not(np.isnan(scores))]
            self.cutoff1 = np.percentile(scores_valid, self.percentile)
            self.cutoff2 = np.percentile(scores_valid, 100 - self.percentile)
        self.logger.info('Score cutoffs: {}-{}'.format(self.cutoff1, self.cutoff2))
        # discretize the values based on the cutoffs
        valid_index = np.nonzero(np.logical_not(np.isnan(scores)))[0]
        scores_valid = scores[valid_index]
        invalid_index = np.logical_and(scores_valid > self.cutoff1, scores_valid < self.cutoff2)
        scores[valid_index[invalid_index]] = np.nan
        valid_index = np.nonzero(np.logical_not(np.isnan(scores)))[0]
        scores_valid = scores[valid_index]
        scores[valid_index[scores_valid <= self.cutoff1]] = 0
        scores[valid_index[scores_valid >= self.cutoff2]] = 1
        fasta_f = IndexedFastaReader(self.sequence_file)
        # calculate base distribution
        self.logger.info('calculate base distribution')
        self.offsets = range(-self.max_offset, self.max_offset + 1)
        base_dist = np.zeros([len(self.offsets), 2, 4], dtype='int64')
        progress = ProgressBar(len(data.names), title='')
        for name in data.names:
            seq = np.frombuffer(fasta_f[name], dtype='S1')
            values = data.feature(self.feature, name)
            ind_valid = (np.logical_not(np.isnan(values)))
            if len(ind_valid) == 0:
                continue
            ind_valid = ind_valid[0]
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
        # plot the base distributions
        if self.plot_type == 'stacked_bar':
            self.stacked_bar(base_dist)
        elif self.plot_type == 'separate_bar':
            self.separate_bar(base_dist)
        # save the table to a text file
        records = []
        for i, j, k in itertools.product(range(len(self.offsets)), range(2), range(4)):
            records.append((self.offsets[i], j, self.alphabet[k], base_dist[i, j, k]))
        df = pd.DataFrame.from_records(records, columns=['Offset', 'State', 'Base', 'Count'])
        table_file = os.path.splitext(self.outfile)[0] + '.txt'
        self.logger.info('save table: {}'.format(table_file))
        df.to_csv(table_file, index=False, sep='\t')

class ScoreDistribution(CommandLineTool):
    description = 'Plot a histogram for values from a GenomicData file'
    arguments = [Argument('score_file', type=str, required=True, help='HDF5 format'),
        Argument('feature', type=str, required=True, help='the feature to plot'),
        Argument('outfile', type=str, required=True, help='the output plot file'),
        Argument('bins', type=int, default=20, help='number of bins'),
        Argument('title', type=str, help='title of the plot')]
    def __call__(self):
        import_matplotlib()

        scores = GenomicData(self.score_file, [self.feature])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(scores.features[self.feature], color='b', edgecolor='k', bins=self.bins)
        ax.set_ylabel('Counts')
        ax.set_xlabel('Scores')
        if self.title is None:
            ax.set_title('Distribution of feature: {}'.format(self.feature))
        else:
            ax.set_title(self.title)
        self.logger.info('save figure: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        plt.savefig(self.outfile)

class RnaplfoldToGenomicData(CommandLineTool):
    description = 'Convert RNAplfold -o files to GenomicData format (HDF5)'
    arguments = [Argument('infile_prefix', type=str, required=True, help='input file names are: <prefix>seqname_basepairs'),
        Argument('sequence_file', type=str, required=True),
        Argument('outfile', short_opt='-o', type=str, required=True),
        Argument('feature', type=str, default='rnaplfold')]
    def __call__(self):
        names = []
        scores = []
        for name, seq in read_fasta(self.sequence_file):
            m = pd.read_table('{}{}_basepairs'.format(self.infile_prefix, name), header=None, sep='\s+')
            bp = np.zeros((len(seq), len(seq)), dtype='float64')
            bp[m[0] - 1, m[1] - 1] = m[2]
            bp += bp.T
            scores.append(bp.sum(axis=0))
            names.append(name)
        self.logger.info('save file: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        GenomicData.from_data(names, features={self.feature: scores},
            create_namedict=False).save(self.outfile)

class RnafoldToGenomicData(CommandLineTool):
    description = 'Convert RNAfold -o files to GenomicData format (HDF5)'
    arguments = [Argument('infile', short_opt='-i', type=str, required=True, help='output file from RNAfold'),
        Argument('outfile', short_opt='-o', type=str, required=True),
        Argument('feature', type=str, default='rnafold')]
    def __call__(self):
        names = []
        scores = []
        for name, seq, structure, energy in read_rnafold(self.infile):
            scores.append((np.frombuffer(structure, dtype='S1') != '.').astype('float32'))
            names.append(name)
        self.logger.info('save file: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        GenomicData.from_data(names, features={self.feature: scores},
            create_namedict=False).save(self.outfile)

class CtToGenomicData(CommandLineTool):
    description = 'Convert structure files in CT format to GenomicData format (HDF5)'
    arguments = [Argument('indir', short_opt='-i', type=str, required=True, help='input directory containing *.ct files'),
        Argument('outfile', short_opt='-o', type=str, required=True),
        Argument('filename_as_name', action='store_true',
            help='use filename (with directory name and extension removed) as the sequence name'),
        Argument('feature', type=str, default='ct')]
    def __call__(self):
        import glob
        names = []
        scores = []
        for ct_file in glob.iglob(self.indir + '/*.ct'):
            name, seq, pairs = read_ct(ct_file)
            if self.filename_as_name:
                name = os.path.splitext(os.path.basename(ct_file))[0]
            names.append(name)
            scores.append((np.asarray(pairs) != 0).astype('float32'))
        self.logger.info('save file: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        GenomicData.from_data(names, features={self.feature: scores},
            create_namedict=False).save(self.outfile)

class CtToFasta(CommandLineTool):
    description = 'Convert structure files in CT format to GenomicData format (HDF5)'
    arguments = [Argument('indir', type=str, required=True, help='input directory containing *.ct files'),
        Argument('outfile', short_opt='-o', type=str, required=True),
        Argument('filename_as_name', action='store_true',
            help='use filename (with directory name and extension removed) as the sequence name'),
        Argument('dna', action='store_true', help='convert U to T')]
    def __call__(self):
        import glob
        self.logger.info('open file: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        f_fasta = open(self.outfile, 'w')
        for ct_file in glob.iglob(self.indir + '/*.ct'):
            name, seq, pairs = read_ct(ct_file)
            if self.filename_as_name:
                name = os.path.splitext(os.path.basename(ct_file))[0]
            seq = seq.upper()
            if self.dna:
                seq = seq.replace('U', 'T')
            else:
                seq = seq.replace('T', 'U')
            f_fasta.write('>{}\n'.format(name))
            f_fasta.write('{}\n'.format(seq))
        f_fasta.close()

class IcshapeToGenomicData(CommandLineTool):
    """Input file format:
        Comments start with #.
        Column 1: sequence name
        Column 2: sequence length
        Column 3: rpkm
        Column 4-end: icSHAPE values (NULL for missing)
    Output file format (HDF5):
        name: sequence names with shape (n_seqs,)
        start: start positions with shape (n_seqs,)
        end: end positions with shape (n_seqs,)
        length: sequence length (n_seqs,)
        rpkm: average rpkm (n_seqs,)
        icshape: values with shape (n_values,)
    """
    description = 'Convert icSHAPE output file to GenomicData format'
    arguments = [Argument('infile', short_opt='-i', type=str, required=True, help='icSHAPE output file'),
        Argument('outfile', short_opt='-o', type=str, required=True)]
    def __call__(self):
        name = []
        length = []
        rpkm = []
        icshape = []
        with open(self.infile, 'r') as fin:
            for lineno, line in enumerate(fin):
                if line.startswith('#'):
                    continue
                fields = line.strip('\t').split()
                name.append(fields[0])
                length.append(int(fields[1]))
                rpkm.append(float(fields[2]))
                icshape.append(np.asarray(map(lambda x: x if x != 'NULL' else np.nan, fields[3:]), dtype='float32'))
                if (lineno > 0) and (length[-1] != len(icshape[-1])):
                    print >>sys.stderr, 'Warning: icSHAPE length not equal to {}'.format(length[-1])

        self.logger.info('save file: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        GenomicData.from_data(np.asarray(name, dtype='S'),
            features={'icshape': icshape},
            meta={'length': np.asarray(length, dtype='int64'),
                  'rpkm': np.asarray(rpkm, dtype='float64')},
            create_namedict=True).save(self.outfile)

class IcshapeRtToGenomicData(CommandLineTool):
    """Input file format:
        Comments start with #. Two consecutive lines for one sequence.
        Column 1: sequence name
        Column 2: sequence length
        Column 3: rpkm
        Column 4-end: base density (Line 1), RT stop count (Line 2)
    Input file format (normalized):
        Comments start with #. Two consecutive lines for one sequence.
        Column 1: sequence name
        Column 2: sequence length
        Column 3: type (baseDensity/RTstop)
        Column 4: rpkm
        Column 5: base_frequency
        Column 6-end: base density (Line 1), RT stop count (Line 2)

    Output file format (HDF5):
        name: sequence names with shape (n_seqs,)
        start: start positions with shape (n_seqs,)
        end: end positions with shape (n_seqs,)
        length: sequence length (n_seqs,)
        rpkm: average rpkm (n_seqs,)
        base_density: values with shape (n_values,)
        rt_stop: values with shape (n_values,)
    """
    description = 'Convert icSHAPE RT-stop file to GenomicData format'
    arguments = [Argument('infile', short_opt='-i', type=str, required=True, help='icSHAPE output file'),
        Argument('outfile', short_opt='-o', type=str, required=True),
        Argument('normalized', action='store_true', help='normalized_rt counts')]
    def __call__(self):
        name = []
        length = []
        rpkm = []
        bd = []
        rt = []
        if self.normalized:
            base_frequency_bd = []
            base_frequency_rt = []
        self.logger.info('read input file: ' + self.infile)
        with open(self.infile, 'r') as fin:
            n_lines = 0
            prev_name = None
            prev_fields = []
            for line in append_extra_line(fin):
                if line.startswith('#'):
                    continue
                n_lines += 1
                fields = line.strip().split('\t')
                if self.normalized:
                    if (len(fields) == 0) or ((fields[0] != prev_name) and (prev_name is not None)):
                        #print prev_fields[0][0]
                        if len(prev_fields) != 2:
                            prev_fields = [fields]
                            prev_name = fields[0]
                            continue
                        name.append(prev_fields[0][0])
                        length.append(prev_fields[0][1])
                        rpkm.append(np.mean(map(float, prev_fields[0][3].split(','))))
                        base_frequency_bd.append(float(prev_fields[0][4]))
                        bd.append(np.asarray(prev_fields[0][5:], dtype='float32'))
                        base_frequency_rt.append(float(prev_fields[1][4]))
                        rt.append(np.asarray(prev_fields[1][5:], dtype='float32'))
                        prev_fields = []

                    prev_name = fields[0]
                    prev_fields.append(fields)
                else:
                    if (len(fields) == 0) or ((fields[0] != prev_name) and (prev_name is not None)):
                        if len(prev_fields) != 2:
                            prev_fields = [fields]
                            prev_name = fields[0]
                            continue
                        name.append(prev_fields[0][0])
                        length.append(int(prev_fields[0][1]))
                        rpkm.append(np.mean(map(float, prev_fields[0][2].split(','))))
                        bd.append(np.asarray(prev_fields[0][3:], dtype='float32'))
                        rt.append(np.asarray(prev_fields[1][3:], dtype='float32'))
                        prev_fields = []

                    prev_name = fields[0]
                    prev_fields.append(fields)

        meta = {'rpkm': np.asarray(rpkm, dtype='float64'),
                'length': np.asarray(length, dtype='int64')}
        if self.normalized:
            meta['base_frequency_bd'] = np.asarray(base_frequency_bd, dtype='float64')
            meta['base_frequency_rt'] = np.asarray(base_frequency_rt, dtype='float64')
        self.logger.info('save file: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        GenomicData.from_data(name,
            features={'base_density': bd, 'rt_stop': rt},
            meta=meta, create_namedict=False).save(self.outfile)

class SubsetGenomicDataByRegion(CommandLineTool):
    description = 'Select a subset of a GenomicData file by regions'
    arguments = [Argument('infile', short_opt='-i', type=str, required=True, help='GenomicData file'),
        Argument('region_file', type=str, required=True,
            help='regions to select in BED format (transcript coordinates)'),
        Argument('outfile', short_opt='-o', type=str, required=True),
        Argument('feature', type=str, required=True)]
    def __call__(self):
        import pandas as pd
        self.data = GenomicData(self.infile, feature_names=[self.feature])
        regions = pd.read_table(self.region_file, header=None)
        outdata = []
        outnames = []
        for row in regions.itertuples(index=None, name=None):
            values = self.data.feature(self.feature, row[0])
            if values is not None:
                outdata.append(values[row[1]:row[2]])
                outnames.append(row[3])
        self.logger.info('save file: {}'.format(self.outfile))
        prepare_output_file(self.outfile)
        GenomicData.from_data(outnames, features={self.feature: outdata}).save(self.outfile)

class NormalizeRtStopByBaseDensity(CommandLineTool):
    arguments = [Argument('input_file', short_opt='-i', type=str, required=True,
            help='Background filein GenomicData format'),
        Argument('output_file', short_opt='-o', type=str, required=True,
            help='output file in GenomicData format')]
    def __call__(self):
        import h5py

        self.logger.info('read input file: ' + self.input_file)
        data = GenomicData(self.input_file)
        normalized_rt_stop = []
        for name in data.names:
            base_density = data.feature('base_density', name).astype(np.float32)
            rt_stop = data.feature('rt_stop', name).astype(np.float32)
            normalized_rt_stop.append((rt_stop + 1)/(base_density + 1))
        self.logger.info('save file: ' + self.output_file)
        prepare_output_file(self.output_file)
        GenomicData.from_data(data.names, features={'rt_stop': normalized_rt_stop}).save(self.output_file)


class CreateDatasetFromGenomicData(CommandLineTool):
    description = 'Create a training dataset from GenomicData format'
    arguments = [Argument('infile', short_opt='-i', type=str, required=True, help='HDF5 format'),
        Argument('sequence_file', type=str, required=True, help='FASTA file'),
        Argument('outfile', short_opt='-o', type=str, required=True),
        Argument('feature', type=str,
            help='feature name to use as target values'),
        Argument('stride', type=int, default=1),
        Argument('window_size', type=int, default=50),
        Argument('cutoff1', type=float),
        Argument('cutoff2', type=float),
        Argument('percentile', type=float, default=5),
        Argument('no_balance', action='store_true'),
        Argument('alphabet', type=str, default='ATCG'),
        Argument('no_shuffle_index', action='store_true'),
        Argument('train_test_split', type=float, default=2,
            help='split the dataset into a training set and test based on the ratio if different from 2'),
        Argument('offset', type=int,
            help='the offset in the sequence to predict. Default is the middle point.'),
        Argument('seed', type=int,
            help='set the random seed for numpy'),
        Argument('dense_output', action='store_true',
            help='predict values for every position in the sequence'),
        Argument('min_coverage', type=float, default=0.5,
            help='minimum proportion of values that is not NaN for each sequence for dense output'),
        Argument('balance_nucleotide', action='store_true',
            help='make the class labels balanced for each nucleotide at the predicted site'),
        Argument('balance_kmer', action='store_true',
            help='make the class labels balanced for each kmer around the predicted site'),
        Argument('kmer_start', type=int, default=0),
        Argument('kmer_end', type=int, default=1),
        Argument('separate', action='store_true')]

    def iter_sequence_target(self, sequence, target):
        if self.offset is None:
            self.offset = (self.window_size + 1)/2
        for i in range(0, len(sequence) - self.window_size, self.stride):
            yield sequence[i:(i + self.window_size)], target[i + self.offset]

    def create_dataset_dense(self, names):
        window_size = self.window_size
        X = []
        y = []
        progress = ProgressBar(len(names), report_freq=10)
        max_nan_count = int((1.0 - self.min_coverage)*window_size)
        for name in names:
            target = self.data.feature(self.feature, name)
            seq = self.sequences[name]
            i = 0
            while i < len(seq) - window_size:
                target_w = target[i:(i + window_size)]
                if np.isnan(target_w).sum() <= max_nan_count:
                    seq_w = seq[i:(i + window_size)]
                    X.append(sequence_to_array(seq_w))
                    y.append(target_w)
                    i += window_size
                else:
                    i += self.stride
            progress.update()
        progress.finish()
        # deal with class imbalance
        n_samples = len(X)
        X = np.concatenate(X).flatten()
        y = np.concatenate(y).flatten()
        if not self.no_balance:
            not_nan_index = np.nonzero(np.logical_not(np.isnan(y)))[0]
            pos_index = not_nan_index[y[not_nan_index] > 0.5]
            neg_index = not_nan_index[y[not_nan_index] < 0.5]
            n_samples_balanced = min(len(pos_index), len(neg_index))
            self.logger.info('number of positive/negative samples in a balanced dataset: %d'%n_samples_balanced)
            if len(pos_index) > len(neg_index):
                np.random.shuffle(pos_index)
                y[pos_index[n_samples_balanced:]] = np.nan
            elif len(pos_index) < len(neg_index):
                np.random.shuffle(neg_index)
                y[neg_index[n_samples_balanced:]] = np.nan
        X = X.reshape((n_samples, window_size, len(self.alphabet)))
        y = y.reshape((n_samples, window_size))
        if not self.no_shuffle_index:
            index = np.arange(n_samples)
            np.random.shuffle(index)
            X = X[index]
            y = y[index]
        return X, y


    def subsample_minor_class(self, X, y):
        ind_pos = np.nonzero(y == 1)[0]
        ind_neg = np.nonzero(y == 0)[0]
        if ind_pos.shape[0] > ind_neg.shape[0]:
            np.random.shuffle(ind_pos)
            ind_pos = ind_pos[:ind_neg.shape[0]]
        elif ind_pos.shape[0] < ind_neg.shape[0]:
            np.random.shuffle(ind_neg)
            ind_neg = ind_neg[:ind_pos.shape[0]]
        ind = np.concatenate([ind_pos, ind_neg])
        return ind

    def create_dataset_single(self, names):
        X = []
        y = []
        progress = ProgressBar(len(names), report_freq=10)
        for name in names:
            for seq, target in self.iter_sequence_target(self.sequences[name],
                    self.data.feature(self.feature, name)):
                if not np.isnan(target):
                    X.append(seq)
                    y.append(target)
            progress.update()
        progress.finish()
        n_samples = len(X)
        # deal with class imbalance
        X = np.frombuffer(''.join(X), dtype='S1').reshape((n_samples, self.window_size))
        X = onehot_encode(X, alphabet=self.alphabet)
        y = np.asarray(y, dtype='int32')

        if not self.no_shuffle_index:
            self.logger.info('shuffle sample indices')
            ind = np.arange(len(y))
            np.random.shuffle(ind)
            X = X[ind]
            y = y[ind]

        def subsample_minor_class(y):
            ind_pos = np.nonzero(y == 1)[0]
            ind_neg = np.nonzero(y == 0)[0]
            if ind_pos.shape[0] > ind_neg.shape[0]:
                np.random.shuffle(ind_pos)
                ind_pos = ind_pos[:ind_neg.shape[0]]
            elif ind_pos.shape[0] < ind_neg.shape[0]:
                np.random.shuffle(ind_neg)
                ind_neg = ind_neg[:ind_pos.shape[0]]
            ind = np.concatenate([ind_pos, ind_neg])
            return ind

        if self.balance_nucleotide:
            Xs = []
            ys = []
            for i_nuc in range(len(self.alphabet)):
                ind = np.nonzero(X[:, self.offset, i_nuc] == 1)[0]
                ind = ind[subsample_minor_class(y[ind])]
                Xs.append(X[ind])
                ys.append(y[ind])
            return Xs, ys
        elif self.balance_kmer:
            Xs = []
            ys = []
            kmer_start = self.kmer_start + self.offset
            kmer_end = self.kmer_end + self.offset
            n = len(self.alphabet)
            k = kmer_end - kmer_start
            factor = np.power(n, np.arange(k))
            factor = factor.reshape((1, k))
            kmer_index = np.sum(X[:, kmer_start:kmer_end].argmax(axis=-1)*factor, axis=1)
            # get all kmer sequences
            all_kmers = np.empty((n**k, k), dtype='int32')
            for i in range(k):
                all_kmers[:, i] = np.tile(np.repeat(np.arange(n), n**(k - i - 1)), n**i)
            alphabet = np.frombuffer(self.alphabet, dtype='S1')
            all_kmers = np.take(alphabet, all_kmers)
            all_kmers = map(lambda i: ''.join(all_kmers[i]), range(all_kmers.shape[0]))
            for i_kmer in range(n**k):
                ind = np.nonzero(kmer_index == i_kmer)[0]
                ind = ind[subsample_minor_class(y[ind])]
                Xs.append(X[ind])
                ys.append(y[ind])
            return Xs, ys, all_kmers
        elif not self.no_balance:
            ind = subsample_minor_class(y)
            X = X[ind]
            y = y[ind]
            return X, y
        else:
            return X, y

    def create_dataset(self, names, dense_output=False):
        if dense_output:
            return self.create_dataset_dense(names)
        else:
            return self.create_dataset_single(names)

    def __call__(self):
        self.sequences = dict(read_fasta(self.sequence_file))
        self.data = GenomicData(self.infile)
        if not self.feature:
            self.feature = self.data.features.keys()[0]
        values = self.data.features[self.feature]

        if (self.cutoff1 is None) and (self.cutoff2 is None):
            values_valid = values[np.logical_not(np.isnan(values))]
            self.cutoff1 = np.percentile(values_valid, self.percentile)
            self.cutoff2 = np.percentile(values_valid, 100 - self.percentile)
        if self.seed:
            self.logger.info('set random seed for numpy: {}'.format(self.seed))
            np.random.seed(self.seed)

        self.logger.info('discretize values with cutoffs: {}-{}'.format(self.cutoff1, self.cutoff2))

        not_nan_ind = np.nonzero(np.logical_not(np.isnan(values)))[0]
        one_ind = not_nan_ind[values[not_nan_ind] >= self.cutoff2]
        zero_ind = not_nan_ind[values[not_nan_ind] <= self.cutoff1]
        values[:] = np.nan
        values[one_ind] = 1
        values[zero_ind] = 0

        def save_dataset(filename, X_train, y_train, names_train,
                X_test, y_test, names_test):
            fout = h5py.File(filename, 'w')
            fout.create_dataset('names_train', data=names_train)
            fout.create_dataset('X_train', data=X_train)
            fout.create_dataset('y_train', data=y_train)
            fout.create_dataset('names_test', data=names_test)
            fout.create_dataset('X_test',  data=X_test)
            fout.create_dataset('y_test',  data=y_test)
            fout.create_dataset('offset', data=self.offset)
            fout.close()

        if 0 < self.train_test_split < 1:
            n_seqs = len(self.data.names)
            n_train = int(n_seqs*self.train_test_split)
            train_ind = np.full(n_seqs, False, dtype='bool')
            train_ind[np.random.choice(n_seqs, size=n_train, replace=False)] = True
            test_ind = np.logical_not(train_ind)
            names_train = self.data.names[train_ind]
            names_test = self.data.names[test_ind]

            if self.balance_kmer:
                Xs_train, ys_train, kmers = self.create_dataset(names_train, self.dense_output)
                Xs_test, ys_test, kmers = self.create_dataset(names_test, self.dense_output)
                for i in range(len(kmers)):
                    self.logger.info('number of training/test set for kmer {}: {}/{}'.format(
                        kmers[i], Xs_train[i].shape[0], Xs_test[i].shape[0]))
                if self.separate:
                    for i in range(len(kmers)):
                        nuc = self.alphabet[i]
                        outfile = os.path.join(self.outfile, kmers[i])
                        self.logger.info('save dataset for nucleotide {}: {}'.format(kmers[i], outfile))
                        prepare_output_file(outfile)
                        save_dataset(outfile, Xs_train[i], ys_train[i], names_train,
                            Xs_test[i], ys_test[i], names_test)
                else:
                    X_train = np.concatenate(Xs_train, axis=0)
                    y_train = np.concatenate(ys_train, axis=0)
                    ind = np.arange(X_train.shape[0])
                    np.random.shuffle(ind)
                    X_train = X_train[ind]
                    y_train = y_train[ind]

                    X_test = np.concatenate(Xs_test, axis=0)
                    y_test = np.concatenate(ys_test, axis=0)
                    ind = np.arange(X_test.shape[0])
                    np.random.shuffle(ind)
                    X_test = X_test[ind]
                    y_test = y_test[ind]
                    self.logger.info('save dataset with balanced nucletide composition: ' + self.outfile)
                    prepare_output_file(self.outfile)
                    save_dataset(self.outfile, X_train, y_train, names_train,
                        X_test, y_test, names_test)
            else:
                X_train, y_train = self.create_dataset(names_train, self.dense_output)
                X_test, y_test = self.create_dataset(names_test, self.dense_output)
                self.logger.info('save dataset: ' + self.outfile)
                prepare_output_file(self.outfile)
                save_dataset(self.outfile, X_train, y_train, names_train,
                    X_test, y_test, names_test)
        else:
            fout = h5py.File(self.outfile)
            X, y = self.create_dataset(self.data.names)
            fout.create_dataset('X', data=X)
            fout.create_dataset('y', data=y)
            fout.close()

class CalculateStructureScoresForIcshape(CommandLineTool):
    description = 'Create a regression dataset'
    arguments = [Argument('background_file', type=str, required=True, help='GenomicData format'),
        Argument('target_file', type=str, required=True, help='GenomicData format'),
        Argument('outfile', short_opt='-o', type=str, required=True),
        Argument('rpkm_cutoff', type=float, default=30.0,
                 help='only keep transcript with RPKM above the cutoff'),
        Argument('method', type=str, default='count_diff',
                 choices=('count_diff', 'icshape_nowinsor', 'background', 'target')),
        Argument('alpha', type=float, default=0.25,
                 help='alpha value for calculating icSHAPE scores'),
        Argument('feature', type=str, help='feature name to use as target values')]
    def __call__(self):
        import scipy

        self.logger.info('read background file: ' + self.background_file)
        background = GenomicData(self.background_file)
        self.logger.info('read target file: ' + self.target_file)
        target = GenomicData(self.target_file)
        common_names = list(set(background.names) & set(target.names))
        # keep transcripts above the cutoff
        self.logger.info('keep transcripts with RPKM > %.2f'%self.rpkm_cutoff)
        kept_names = []
        tr = [] # target RT
        tb = [] # target base density
        br = [] # background RT
        bb = [] # background base density
        rpkm = []
        length = []

        # smoothing window
        for name in common_names:
            filters = [(background.feature('rpkm', name) > self.rpkm_cutoff),
                       (target.feature('rpkm', name) > self.rpkm_cutoff)]
            if all(filters):
                kept_names.append(name)
                tr.append(target.feature('rt_stop', name))
                tb.append(target.feature('base_density', name))
                br.append(background.feature('rt_stop', name))
                bb.append(background.feature('base_density', name))
                length.append(len(target.feature('rt_stop', name)))
                rpkm.append(background.feature('rpkm', name))
        tr = np.concatenate(tr)
        tb = np.concatenate(tb)
        br = np.concatenate(br)
        bb = np.concatenate(bb)
        rpkm = np.asarray(rpkm)
        length = np.asarray(length, dtype='int64')
        end = np.cumsum(length)
        start = end - length
        # filter by values
        kept_filter = np.logical_not(np.any(np.vstack([np.isnan(tr), np.isnan(tb),
                                                       np.isnan(br), np.isnan(bb),
                                                       tb < 10, bb < 10,
                                                       tr < 2, br < 2]), axis=0))
        self.logger.info('use method: %s'%self.method)
        if self.method == 'count_diff':
            scores = (tr[kept_filter] - br[kept_filter])/(bb[kept_filter])
        elif self.method == 'icshape_nowinsor':
            alpha = 0.25
            scores = (tr[kept_filter] - alpha*br[kept_filter]) / (bb[kept_filter])
        elif self.method == 'background':
            scores = br[kept_filter] / bb[kept_filter]
        elif self.method == 'target':
            scores = tr[kept_filter] / tb[kept_filter]
        else:
            raise ValueError('unknown method: %s'%self.method)
        # winsorization
        lower_cutoff = np.percentile(scores, 1)
        upper_cutoff = np.percentile(scores, 99)
        self.logger.info('winsorize: (%f, %f)'%(lower_cutoff, upper_cutoff))
        scores = np.clip(scores, lower_cutoff, upper_cutoff)
        scores_all = np.full(tr.shape[0], np.nan)
        scores_all[kept_filter] = scores
        scores = [scores_all[(start[i]):(end[i])] for i in range(len(length))]
        self.logger.info('save structure scores to file: ' + self.outfile)
        prepare_output_file(self.outfile)
        GenomicData.from_data(kept_names,
                              features={'scores': scores},
                              meta={'rpkm': rpkm}).save(self.outfile)

class CreateRegressionDatasetForIcshape(CommandLineTool):
    description = 'Create a training dataset from GenomicData format'
    arguments = [Argument('infile', short_opt='-i', type=str, required=True,
                          help='a dataset in GenomicData format'),
                 Argument('sequence_file', type=str, required=True, help='FASTA file'),
                 Argument('outfile', short_opt='-o', type=str, required=True),
                 Argument('window_size', type=int, default=100),
                 Argument('stride', type=int, default=1,
                          help='distance between neighbor windows'),
                 Argument('min_coverage', type=float, default=0.5),
                 Argument('test_size', type=float, default=0.1),
                 Argument('feature', type=str, default='icshape')]

    def create_dataset(self, names):
        from tqdm import tqdm
        window_size = self.window_size
        X = []
        y = []
        max_nan_count = int((1.0 - self.min_coverage) * window_size)
        for name in tqdm(names):
            target = self.data.feature(self.feature, name)
            seq = self.sequences[name]
            i = 0
            while i < len(seq) - window_size:
                target_w = target[i:(i + window_size)]
                if np.isnan(target_w).sum() <= max_nan_count:
                    seq_w = seq[i:(i + window_size)]
                    X.append(sequence_to_array(seq_w))
                    y.append(target_w)
                i += self.stride
        n_samples = len(X)
        X = np.concatenate(X).reshape((n_samples, -1, 4))
        y = np.concatenate(y).reshape((n_samples, -1))
        return X, y

    def __call__(self):
        from sklearn.model_selection import train_test_split
        import h5py

        self.logger.info('read sequence file: ' + self.sequence_file)
        self.sequences = dict(read_fasta(self.sequence_file))
        self.logger.info('read input file: ' + self.infile)
        self.data = GenomicData(self.infile)

        names_train, names_test = train_test_split(self.data.names, test_size=self.test_size)
        self.logger.info('create training dataset')
        X_train, y_train = self.create_dataset(names_train)
        self.logger.info('create test dataset')
        X_test, y_test = self.create_dataset(names_test)
        self.logger.info('save dataset to file: ' + self.outfile)
        prepare_output_file(self.outfile)
        with h5py.File(self.outfile, 'w') as f:
            f.create_dataset('names_train', data=names_train)
            f.create_dataset('names_test', data=names_test)
            f.create_dataset('X_train', data=X_train)
            f.create_dataset('y_train', data=y_train)
            f.create_dataset('X_test', data=X_test)
            f.create_dataset('y_test', data=y_test)


class CorrelationBetweenIcshape(CommandLineTool):
    description = 'Draw distribution of correlations between icSHAPE values in two dataset'
    arguments = [Argument('infile1', type=str, required=True, help='icSHAPE data in HDF5 format'),
        Argument('infile2', type=str, required=True, help='icSHAPE data in HDF5 format'),
        Argument('transcript_anno', type=str, help='transcript annotation in BED format'),
        Argument('name1', type=str, default='Dataset 1', help='name of dataset 1'),
        Argument('name2', type=str, default='Dataset 2', help='name of dataset 2'),
        Argument('prefix', type=str, required=True, help='plot file with a text table'),
        Argument('feature', type=str, default='icshape', help='feature name to use in the data'),
        Argument('strip_transcript_version', action='store_true', help='drop version number in the transcript names before intersection'),
        Argument('title', type=str, default='Correlation between two icSHAPE datasets', help='title of the plot')
        ]
    def __call__(self):
        from scipy.stats import pearsonr
        import pandas as pd
        import_matplotlib()

        data1 = GenomicData(self.infile1, [self.feature])
        data2 = GenomicData(self.infile2, [self.feature])
        if self.strip_transcript_version:
            self.logger.info('strip version number in the transcript names')
            data1.set_names([name.split('.')[0] for name in data1.names])
            data2.set_names([name.split('.')[0] for name in data2.names])
        common_names = list(set(data1.names) & set(data2.names))
        r = np.zeros(len(common_names), dtype='float64')
        p = np.zeros(len(common_names), dtype='float64')
        for i, name in enumerate(common_names):
            x1 = data1.feature(self.feature, name)
            x2 = data2.feature(self.feature, name)
            ind_valid = np.nonzero(np.logical_and(np.logical_not(np.isnan(x1)),
                np.logical_not(np.isnan(x2))))[0]
            if len(ind_valid) > 2:
                r[i], p[i] = pearsonr(x1[ind_valid], x2[ind_valid])
        r[np.isnan(r)] = 0

        # save correlation coefficients to a table
        df = pd.DataFrame.from_dict({'name': common_names, 'r': r, 'p_value': p})
        outdir = os.path.dirname(self.prefix)
        table_file = self.prefix + '.txt'
        self.logger.info('save text table: {}'.format(table_file))
        make_dir(outdir)
        df.to_csv(table_file, index=False, sep='\t')
        # plot distribution of correlation coefficients
        fig, ax = plt.subplots()
        ax.hist(r, bins=50)
        ax.set_title(self.title)
        self.logger.info('save plot: {}'.format(self.prefix + '.pdf'))
        plt.savefig(self.prefix + '.pdf')
        plt.close(fig)
        # plot distribution of correlation coefficients by strand
        if self.transcript_anno:
            transcript = pd.read_table(self.transcript_anno, header=None)
            # strip transcipt version
            if self.strip_transcript_version:
                transcript.index = transcript[3].str.extract('([^\.]+)\.[0-9]+', expand=False)
            else:
                transcript.index = transcript[3]
            strand = transcript.ix[common_names, 5]

            fig, ax = plt.subplots()
            ax.hist(r[(strand == '+').values], bins=50)
            ax.set_title(self.title + ' (plus strand)')
            self.logger.info('save plot: {}.plus.pdf'.format(self.prefix))
            plt.savefig(self.prefix + '.plus.pdf')
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.hist(r[(strand == '-').values], bins=50)
            ax.set_title(self.title + ' (minus strand)')
            self.logger.info('save plot: {}.minus.pdf'.format(self.prefix))
            plt.savefig(self.prefix + '.minus.pdf')
            plt.close(fig)
        # plot distribution of icSHAPE values of transcripts with low correlation
        ind_low_r = np.nonzero(np.absolute(r) < 0.05)[0]
        low_r_file = self.prefix + '.low_r.pdf'
        self.logger.info('plot transcripts with low correlation: {}'.format(low_r_file))
        with PdfPages(low_r_file) as pdf:
            n_plots = 0;
            for i in ind_low_r:
                fig, ax = plt.subplots(figsize=(12, 2))
                name = common_names[i]
                y1 = data1.feature(self.feature, name)
                y2 = data2.feature(self.feature, name)
                y1[np.isnan(y1)] = -0.1
                y2[np.isnan(y2)] = -0.1
                length = min(300, len(y1))
                x = np.arange(length)
                ax.plot(x, y1[:length], 'b-', label=self.name1)
                ax.plot(x, y2[:length], 'k-', label=self.name2)
                ax.set_title('{} (r={})'.format(name, r[i]))
                ax.set_ylim(-0.2, 1)
                ax.legend()
                n_plots += 1
                if n_plots >= 20:
                    break
                pdf.savefig(fig)
                plt.close(fig)

class GenerateMutatedSequences(CommandLineTool):
    description = 'Generate sequences by mutating sequences one nucleotide at a time from a FASTA file'
    arguments = [Argument('infile', short_opt='-i', type=str, required=True),
        Argument('outfile', short_opt='-o', type=str, required=True)]
    def __call__(self):
        mut_table = {'A': 'TCG', 'T': 'ACG', 'C': 'ATG', 'G': 'ATC'}
        self.logger.info('save output file: ' + self.outfile)
        prepare_output_file(self.outfile)
        fout = open(self.outfile, 'w')
        for name, seq in read_fasta(self.infile):
            fout.write('>%s:WT\n'%(name))
            fout.write(seq)
            fout.write('\n')
            for i in range(len(seq)):
                a = seq[i]
                for b in mut_table[a]:
                    new_seq = bytearray(seq)
                    new_seq[i] = b
                    fout.write('>%s:%s%d%s\n'%(name, a, i + 1, b))
                    fout.write(str(new_seq))
                    fout.write('\n')
        fout.close()

class ExtractRdat(CommandLineTool):
    description = 'Extract sequences and structures from RDAT files'
    arguments = [Argument('id_file', short_opt='-i', type=str, required=True,
            help='a file containing RMDB IDs'),
        Argument('rmdb_dir', type=str, required=True,
            help='directory containing .rdat files'),
        Argument('format', type=str, choices=('fasta', 'rnafold', 'genomic_data')),
        Argument('unique', action='store_true',
            help='only output unique sequences'),
        Argument('outfile', short_opt='-o', type=str, required=True)]

    def get_constructs(self):
        import rdatkit
        with open(self.id_file, 'r') as f:
            ids = f.read().strip().split()
        fout = sys.stdout
        for id in ids:
            rdat = rdatkit.RDATFile()
            with open(os.path.join(self.rmdb_dir, '%s.rdat'%id), 'r') as f:
                rdat.load(f)
            for construct_id, name in enumerate(rdat.constructs):
                yield id, construct_id, name, rdat.constructs[name]

    def write_rnafold(self, fout):
        fout = open(self.outfile, 'w')
        unique_sequences = set()
        for id, construct_id, name, construct in self.get_constructs():
            if len(construct.structure) == 0:
                continue
            if 'X' in construct.sequence:
                continue
            if all(map(lambda x: x == '.', construct.structure)):
                continue
            seqname = '%s:%d %s'%(id, construct_id, name)
            seqpos = map(lambda x: x - construct.offset - 1, construct.seqpos)
            sequence = construct.sequence.upper()
            sequence = sequence.replace('U', 'T')
            #sequence = ''.join(map(lambda i: construct.sequence[i], seqpos)).upper()
            if self.unique:
                if sequence in unique_sequences:
                    continue
                else:
                    unique_sequences.add(sequence)
            structure = construct.structure.upper()
            #structure = ''.join(map(lambda i: construct.structure[i], seqpos)).upper()
            structure = structure.replace('[', '(')
            structure = structure.replace(']', ')')
            fout.write('>%s\n'%seqname)
            fout.write(sequence + '\n')
            fout.write(structure + '\n')
        fout.close()

    def write_genomic_data(self):
        from genomic_data import GenomicData
        import re
        mutation_pat = re.compile(r'([AUGCT])([0-9]+)([AUGCT])')
        seqnames = []
        data = []
        sequences = []
        for id, construct_id, name, construct in self.get_constructs():
            self.logger.info('construct: %s:%s'%(id, construct_id))
            if 'X' in construct.sequence:
                continue
            seqpos = np.asarray(construct.seqpos)
            seqpos -= construct.offset + 1
            for i, section in enumerate(construct.data):
                mutation = section.annotations.get('mutation')
                if not mutation:
                    raise ValueError('mutation annotation not found in %s:%s'%(id, construct_id))
                mutation = mutation[0]
                sequence = np.copy(np.frombuffer(construct.sequence.upper(), dtype='S1'))
                if mutation != 'WT':
                    m = mutation_pat.search(mutation)
                    if not m:
                        raise ValueError('mutation information not found for %s:%s:%s'%(id, construct_id, mutation))
                    old_base, mutpos, new_base = m.group(1), m.group(2), m.group(3)
                    mutpos = int(mutpos) - construct.offset - 1
                    if sequence[mutpos] != old_base:
                        raise ValueError('mutated base is not the same as the original sequence at %s:%s:%s'%(id, construct_id, mutation))
                    sequence[mutpos] = new_base
                sequence[sequence == 'U'] = 'T'
                sequence = sequence[seqpos]
                sequences.append(sequence)
                seqnames.append('%s:%d:%s %s'%(id, construct_id,  mutation, name))
                data.append(np.asarray(section.values, dtype='float32'))
        self.logger.info('save file: ' + self.outfile)
        prepare_output_file(self.outfile)
        GenomicData.from_data(seqnames, features={'reactivity': data, 'sequence': sequences}).save(self.outfile)

    def __call__(self):
        if self.format == 'rnafold':
            self.write_rnafold()
        elif self.format == 'genomic_data':
            self.write_genomic_data()

class RNAfoldBasePairProfile(CommandLineTool):
    description = 'Calculate 1D base-pair probabilities using the RNAplfold algorithm'
    arguments = [Argument('input_file', short_opt='-i', type=str, required=True,
                          help='a sequence file in FASTA format'),
                 Argument('names_file', type=str,
                          help='an HDF5 file containing a list of sequence names (format: filename:dataset'),
                 Argument('window_size', type=int, default=400),
                 Argument('feature', type=str, default='bp_profile',
                          help='feature name in the output file'),
                 Argument('n_jobs', short_opt='-j', type=int, default=1,
                          help='number of parallel jobs'),
                 Argument('batch_size', type=int, default=100,
                          help='number of sequences to process in each batch'),
                 Argument('output_file', short_opt='-o', type=str, required=True,
                          help='output file in GenomicData format')]
    def __call__(self):
        from ViennaRNA import unpaired_prob_local
        import numpy as np
        from tqdm import tqdm
        from formats import read_hdf5_dataset
        from joblib import Parallel, delayed

        self.logger.info('read input sequence file: ' + self.input_file)
        sequences = dict(read_fasta(self.input_file))
        if self.names_file is not None:
            sel_names = read_hdf5_dataset(self.names_file)
            sequences = {name:sequences[name] for name in sel_names}
        probs = []
        names = []
        self.logger.info('calculate base-pair profiles')
        if self.n_jobs > 1:
            generator = sequences.iteritems()
            end_loop = False
            with Parallel(n_jobs=self.n_jobs) as parallel:
                pbar = tqdm(total=len(sequences))
                while not end_loop:
                    jobs = []
                    try:
                        for i in range(self.batch_size):
                            name, seq = generator.next()
                            names.append(name)
                            jobs.append(delayed(unpaired_prob_local)(seq, window_size=self.window_size, ulength=1))
                    except StopIteration:
                        end_loop = True
                    probs += parallel(jobs)
                    pbar.update(len(jobs))
                pbar.close()
            for i in range(len(probs)):
                probs[i] = 1.0 - np.ravel(probs[i])
        else:
            for name, seq in tqdm(sequences.iteritems(), total=len(sequences)):
                seq = seq.replace('T', 'U')
                names.append(name)
                prob = 1.0 - unpaired_prob_local(seq, window_size=self.window_size, ulength=1)
                probs.append(np.ravel(prob.astype('float32')))
        self.logger.info('save data to file: ' + self.output_file)
        prepare_output_file(self.output_file)
        GenomicData.from_data(names, features={self.feature: probs}).save(self.output_file)

if __name__ == '__main__':
    CommandLineTool.from_argv()()
