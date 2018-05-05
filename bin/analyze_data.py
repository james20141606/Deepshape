#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')

def plot_fft(signals, title):
    N = signals.shape[0]
    yf = fft(signals)
    fig, ax = plt.subplots(figsize=(18, 2))
    period = float(N)/np.arange(1, N//2 + 1)
    ax.plot(period[:1:-1], np.abs(yf[0:(N//2)])[:1:-1])
    ax.set_xlim(1, 10)
    ax.set_xlabel('Period/nt')
    ax.set_ylabel('Intensity')
    ax.set_title(title)

def analyze_nucleotide_periodicity(args):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    sns.set()
    from scipy.fftpack import fft
    from formats import read_fasta
    from ioutils import prepare_output_file

    logger.info('read sequence file: ' + args.input_file)
    sequences = {name:np.frombuffer(seq, dtype='S1') for name, seq in read_fasta(args.input_file)}
    aligned_length = args.aligned_length
    alphabet = args.alphabet

    def calc_nucleotide_freq(sequences, direction, alphabet='ATCG', aligned_length=100):
        alphabet = np.frombuffer(alphabet, dtype='S1')
        m = np.full((len(sequences), aligned_length), 'N', dtype='S1')
        for i, name in enumerate(sequences.keys()):
            x = sequences[name]
            L = min(x.shape[0], aligned_length)
            if direction == '5p':
                m[i, :L] = x[:L]
            elif direction == '3p':
                m[i, -L:] = x[-L:]
        transcript_counts = np.sum(m != 'N', axis=0)
        m_onehot = (m[:, :, np.newaxis] == alphabet[np.newaxis, np.newaxis, :])
        m_counts = np.sum(m_onehot, axis=0).astype(np.float64)
        m_freq = m_counts/np.sum(m_counts, axis=1)[:, np.newaxis]
        return m_freq

    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    with PdfPages(args.output_file) as pdf:
        # 5'-end
        nucleotide_freq_5p = calc_nucleotide_freq(sequences, '5p',
            alphabet=alphabet, aligned_length=aligned_length)
        fig, ax = plt.subplots(figsize=(18, 4))
        for i, nucleotide in enumerate(alphabet):
            ax.plot(np.arange(aligned_length), nucleotide_freq_5p[:, i], label=nucleotide)
        ax.set_xlabel('Position in CDS from 5\'-end')
        ax.set_ylabel('Nucleotide frequency')
        ax.set_xlim(0, aligned_length)
        ax.set_ylim(0, 1)
        plt.legend()
        pdf.savefig()
        plt.close()

        # 3'-end
        nucleotide_freq_3p = calc_nucleotide_freq(sequences, '3p',
            alphabet=alphabet, aligned_length=aligned_length)
        fig, ax = plt.subplots(figsize=(18, 4))
        for i, nucleotide in enumerate(alphabet):
            ax.plot(np.arange(-aligned_length, 0), nucleotide_freq_3p[:, i], label=nucleotide)
        ax.set_xlabel('Distance from CDS from 3\'-end')
        ax.set_ylabel('Nucleotide frequency')
        ax.set_xlim(-aligned_length, 0)
        ax.set_ylim(0, 1)
        plt.legend()
        pdf.savefig()
        plt.close()

        # FFT
        for i, nucleotide in enumerate(alphabet):
            plot_fft(nucleotide_freq_5p[:, i], 'Nucleotide %s from 5\'-end'%nucleotide)
            pdf.savefig()
            plt.close()
        for i, nucleotide in enumerate(alphabet):
            plot_fft(nucleotide_freq_3p[:, i], 'Nucleotide %s from 3\'-end'%nucleotide)
            pdf.savefig()
            plt.close()

def analyze_reactivity_periodicity(args):
    import h5py
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    sns.set()
    from genomic_data import GenomicData
    from scipy.fftpack import fft
    from formats import read_fasta
    from ioutils import prepare_output_file

    logger.info('read input file: ' + args.input_file)
    reactivities = {}
    if args.assay_type == 'shapemap':
        with h5py.File(args.input_file, 'r') as f:
            for tx_id in f['reactivities'].keys():
                reactivities[tx_id] = f['reactivities/' + tx_id][:]
    elif args.assay_type in ('icshape', 'rt_stop'):
        data = GenomicData(args.input_file)
        for name in data.names:
            reactivities[name] = data.feature(args.assay_type, name)
    seq_names = np.asarray(reactivities.keys(), dtype='S')
    # get coverage of each transcript
    lengths = np.asarray([reactivities[name].shape[0] for name in seq_names])
    coverage = np.asarray([np.sum(~np.isnan(reactivities[name])) for name in seq_names])
    coverage = coverage.astype(np.float64)/lengths
    # filter sequences of by coverage
    seq_names = seq_names[coverage >= args.min_coverage]
    reactivities = {name:reactivities[name] for name in seq_names}

    def calc_average_reactivities(reactivities, direction, aligned_length=100):
        reactivities_avg = np.full((len(reactivities), aligned_length), np.nan)
        for i, name in enumerate(reactivities.keys()):
            x = reactivities[name]
            L = min(x.shape[0], aligned_length)
            if direction == '5p':
                reactivities_avg[i, :L] = x[:L]
            elif direction == '3p':
                reactivities_avg[i, -L:] = x[-L:]
        transcript_counts = np.sum(~np.isnan(reactivities_avg), axis=0)
        reactivities_avg = np.nan_to_num(reactivities_avg)
        reactivities_avg = np.sum(reactivities_avg, axis=0)/transcript_counts.astype(np.float64)
        return reactivities_avg

    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    aligned_length = args.aligned_length
    with PdfPages(args.output_file) as pdf:
        # 5'-end
        reactivities_avg_5p = calc_average_reactivities(reactivities, '5p',
            aligned_length=aligned_length)
        fig, ax = plt.subplots(figsize=(18, 4))
        ax.plot(np.arange(aligned_length), reactivities_avg_5p)
        ax.set_xlabel('Position in CDS from 5\'-end')
        ax.set_ylabel('Reactivity')
        ax.set_xlim(0, aligned_length)
        pdf.savefig()
        plt.close()

        # 3'-end
        reactivities_avg_3p = calc_average_reactivities(reactivities, '3p',
            aligned_length=aligned_length)
        fig, ax = plt.subplots(figsize=(18, 4))
        ax.plot(np.arange(-aligned_length, 0), reactivities_avg_3p)
        ax.set_xlabel('Distance from CDS from 3\'-end')
        ax.set_ylabel('Reactivity')
        ax.set_xlim(-aligned_length, 0)
        pdf.savefig()
        plt.close()

        ## FFT
        plot_fft(reactivities_avg_5p, 'CDS from 5\'-end')
        pdf.savefig()
        plt.close()
        plot_fft(reactivities_avg_3p, 'CDS from 3\'-end')
        pdf.savefig()
        plt.close()

def analyze_periodicity(args):
    import h5py
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import seaborn as sns
    sns.set()
    from genomic_data import GenomicData

    logger.info('read input file: ' + args.input_file)
    reactivities = {}
    sequences = {}
    if args.assay_type == 'shapemap':
        with h5py.File(args.input_file, 'r') as f:
            for tx_id in f['rep1'].keys():
                reactivities[tx_id] = f['rep1/' + tx_id][:]
            for tx_id in f['seq'].keys():
                sequences[tx_id] = f['seq/' + tx_id][()]
    elif args.assay_type == 'icshape':
        icshape = GenomicData(args.input_file)
        for name in icshape.names:
            reactivities[name] = icshape.feature('icshape', name)
        for name, seq in read_fasta(args.sequence_file):
            if name in icshape.names:
                sequences[name] = np.frombuffer(seq, dtype='S1')
    seq_names = sequences.keys()

    reactivities_concat = np.concatenate([reactivities[name] for name in seq_names])
    sequences_concat = np.concatenate([sequences[name] for name in seq_names])
    notnan_mask = ~np.isnan(reactivities_concat)

    # plot overall distribution of SHAPE reactivity
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(reactivities_concat[notnan_mask], bins=50)
    ax.set_xlabel('Reactivity')
    ax.set_ylabel('Counts')
    plt.savefig()

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Analyze data')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('reactivity_periodicity',
                                   help='Analyze various structure profiling data')
    parser.add_argument('--input-file', '-i', type=str, required=True,
            help='input structure profiling file')
    parser.add_argument('--assay-type', '-a', type=str, required=True,
            choices=('icshape', 'shapemap', 'rt_stop'))
    parser.add_argument('--output-file', '-o', type=str, required=True,
            help='output plot in PDF format')
    parser.add_argument('--aligned-length', '-l', type=int, default=100)
    parser.add_argument('--min-coverage', type=float, default=0.5)

    parser = subparsers.add_parser('nucleotide_periodicity',
                    help='Analyze nucleotide periodicity in CDS regions')
    parser.add_argument('--input-file', '-i', type=str, required=True,
            help='sequence in FASTA format')
    parser.add_argument('--output-file', '-o', type=str, required=True,
            help='output plot in PDF format')
    parser.add_argument('--aligned-length', '-l', type=int, default=100)
    parser.add_argument('--alphabet', '-a', type=str, default='ATGC')

    args = main_parser.parse_args()

    args = main_parser.parse_args()
    logger = logging.getLogger('analyze_data.' + args.command)

    import numpy as np
    from scipy.fftpack import fft
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    if args.command == 'reactivity_periodicity':
        analyze_reactivity_periodicity(args)
    elif args.command == 'nucleotide_periodicity':
        analyze_nucleotide_periodicity(args)