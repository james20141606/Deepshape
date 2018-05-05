#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')

def get_te_icshape(args):
    import numpy as np
    import h5py
    import pandas as pd
    from genomic_data import GenomicData

    logger.info('read TE region file: ' + args.bed_file)
    bed = pd.read_table(args.bed_file, header=None)
    bed[3] = bed[3].astype('S')

    logger.info('read icSHAPE data file: ' + args.icshape_file)
    icshape = GenomicData(args.icshape_file)

    logger.info('create output file: ' + args.output_file)
    te_data = []
    te_names = []
    for row in bed.itertuples(index=False):
        data = icshape.feature(args.feature, row[0])
        if data is not None:
            te_data.append(data[row[1]:row[2]])
            te_names.append('%s,%s,%d,%d'%(row[3], row[0], row[1], row[2]))
    
    logger.info('create output file: ' + args.output_file)
    GenomicData.from_data(names=te_names, features={args.feature: te_data}).save(args.output_file)
    
def plot_alignment(args):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    from Bio import AlignIO

    logger.info('read alignment file: ' + args.input_file)
    try:
        alignment = AlignIO.read(open(args.input_file), 'stockholm')
    except ValueError:
        logger.error('exit because the input file contains no records')

    logger.info("Alignment length %i" % alignment.get_alignment_length())
    align_array = np.array([list(rec.upper()) for rec in alignment], np.character)
    n_seqs, n_columns = align_array.shape
    logger.info('Number of alignments: %d'%n_seqs)
    alphabet = np.array(['-', 'A', 'C', 'G', 'T'], dtype=np.character)
    NucleotideColorMap = matplotlib.colors.ListedColormap(np.array([[1.0, 1.0, 1.0],
                                                       [1.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0],
                                                       [1.0, 0.65, 0.0],
                                                       [0.0, 0.0, 1.0]]))
    # onehot coding of the alignment, shape: (n_seqs, n_columns, alphabet_size)
    align_onehot = align_array[:, :, np.newaxis] == alphabet[np.newaxis, np.newaxis, :]
    # integer coding of the alignment, shape: (n_seqs, n_columns)
    align_integer = np.argmax(align_onehot, axis=-1)
    # plot sequence alignment
    fig, ax = plt.subplots(figsize=(18, 3))
    sns.heatmap(align_integer, cmap=NucleotideColorMap,
                    norm=matplotlib.colors.Normalize(),
                    xticklabels=False, yticklabels=False, ax=ax, cbar=False)
    logger.info('create alignment plot file: ' + args.output_file)
    plt.savefig(args.output_file)
     
def hmm_align_reactivities(args):
    import numpy as np
    from Bio import AlignIO
    from glob import glob
    import re
    import h5py
    from tqdm import tqdm
    from genomic_data import GenomicData

    logger.info('read reactivity file: ' + args.reactivity_file)
    data = GenomicData(args.reactivity_file)
    reactivities = {name:data.feature(args.feature, name) for name in data.names}

    logger.info('read alignment directory: ' + args.alignment_dir)
    pat_id = re.compile(r'[0-9]*\|*(?P<te_type>[^:]+)::(?P<seq_name>[^:]+):(?P<start>[0-9]+)-(?P<end>[0-9]+)')
    logger.info('create output file: ' + args.output_file)
    fout = h5py.File(args.output_file, 'w')

    for alignment_file in tqdm(glob(os.path.join(args.alignment_dir, '*.sto')), unit='file'):
        with open(alignment_file, 'r') as fin:
            reactivities_tx = []
            sequences_tx = []
            seq_names = []
            for record in AlignIO.read(fin, 'stockholm'):
                # parse transcriptomic coordinates from sequence names
                m = pat_id.match(record.id)
                if m is None:
                    raise ValueError('invalid record name %s in file: %s'%(record.id, alignment_file))
                r = reactivities.get(m.group('seq_name'))
                if r is not None:
                    r_aligned = np.full(len(record.seq), np.nan, dtype=np.float32)
                    # map reactivities to alignment
                    seq = np.frombuffer(str(record.seq), dtype='S1')
                    r_aligned[seq != '-'] = r[int(m.group('start')):int(m.group('end'))]
                    if np.all(np.isnan(r_aligned)):
                        continue
                    reactivities_tx.append(r_aligned.reshape((1, -1)))
                    sequences_tx.append(seq.reshape((1, -1)))
                    seq_names.append(record.id)
            if len(reactivities_tx) >= args.min_records:
                g = fout.create_group(os.path.splitext(os.path.basename(alignment_file))[0])
                g.create_dataset('reactivities', data=np.concatenate(reactivities_tx, axis=0))
                g.create_dataset('sequences', data=np.concatenate(sequences_tx, axis=0))
                g.create_dataset('seq_names', data=np.asarray(seq_names))
    fout.close()

def plot_hmm_align_reactivities(args):
    import numpy as np
    import h5py
    from tqdm import tqdm
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    sns.set()

    alphabet = np.array(['-', 'A', 'C', 'G', 'T'], dtype=np.character)
    NucleotideColorMap = matplotlib.colors.ListedColormap(np.array([[1.0, 1.0, 1.0],
                                                       [1.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0],
                                                       [1.0, 0.65, 0.0],
                                                       [0.0, 0.0, 1.0]]))

    logger.info('read input reactivities file: ' + args.input_file)
    reactivities = {}
    sequences = {}
    seq_names = {}
    with h5py.File(args.input_file, 'r') as fin:
        for te_type in fin.keys():
            reactivities[te_type] = fin[te_type + '/reactivities'][:]
            sequences[te_type] = fin[te_type + '/sequences'][:]
            seq_names[te_type] = fin[te_type + '/seq_names'][:]
    logger.info('create output plot file: ' + args.output_file)
    with PdfPages(args.output_file) as pdf:
        for te_type in tqdm(reactivities.keys(), unit=('te_type')):
            fig, axes = plt.subplots(2, 1, figsize=(20, 1 + 0.2*len(reactivities[te_type])))
            # plot reactivities
            sns.heatmap(reactivities[te_type], 
                xticklabels=False, yticklabels=seq_names[te_type], cbar=True, 
                ax=axes[0], cmap=matplotlib.cm.RdBu_r)
            axes[0].set_title('Reactivities')
            # plot sequences
            # onehot coding of the alignment, shape: (n_seqs, n_columns, alphabet_size)
            sequences_te_onehot = (sequences[te_type][:, :, np.newaxis] == alphabet[np.newaxis, np.newaxis, :])
            # integer coding of the alignment, shape: (n_seqs, n_columns)
            sequences_te_integer = np.argmax(sequences_te_onehot, axis=-1)
            sns.heatmap(sequences_te_integer, cmap=NucleotideColorMap,
                    norm=matplotlib.colors.NoNorm(),
                    xticklabels=False, yticklabels=seq_names[te_type], ax=axes[1])
            axes[1].set_title('Sequences')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

def hmm_align_stats(args):
    import numpy as np
    from Bio import AlignIO

    logger.info('read alignment file: ' + args.input_file)
    try:
        alignment = AlignIO.read(open(args.input_file), 'stockholm')
    except ValueError:
        logger.error('exit because the input file contains no records')
        return
    name = os.path.splitext(os.path.basename(args.input_file))[0]

    logger.info('Alignment name: ' + name)
    logger.info("Alignment length %i" % alignment.get_alignment_length())
    align_array = np.array([list(rec.upper()) for rec in alignment], np.character)
    n_seqs, n_columns = align_array.shape
    logger.info('Number of alignments: %d'%n_seqs)
    alphabet = np.array(['-', 'A', 'C', 'G', 'T'], dtype=np.character)
    
    # onehot coding of the alignment, shape: (n_seqs, n_columns, alphabet_size)
    align_onehot = align_array[:, :, np.newaxis] == alphabet[np.newaxis, np.newaxis, :]
    # integer coding of the alignment, shape: (n_seqs, n_columns)
    align_integer = np.argmax(align_onehot, axis=-1)
    # occureneces of each residual in each column, shape: (n_columns, alphabet_size)
    occurrences = np.sum(align_onehot, axis=0)
    # number of distinct residual types in each column, shape: (n_columns)
    n_residual_types = np.sum(occurrences > 0, axis=1)
    # normalized occurrences of each residual in each column
    frequencies = occurrences.astype(np.float64)/n_seqs
    # fraction of gaps in each column, shape: (n_columns)
    gaps = np.sum(align_onehot[:, :, 0], axis=0).astype(np.float64)/n_seqs
    # fraction of residuals excluding gaps('-') in each column, shape: (n_columns)
    occupancies = 1.0 - gaps
    # most frequent residuals in each column, shape: (n_columns,)
    consensus = alphabet[np.argmax(occurrences, axis=1)]
    # number of residuals in each sequence that is different from the consensus
    n_mutations = np.sum(align_array != consensus[np.newaxis, :], axis=1)
    # proportion of same residuals as consensus sequences in each column, shape: (n_columns)
    # conservation = (number of most frequent residual) / (total number of sequences)
    conservation = np.max(occurrences, axis=1).astype(np.float32)/n_seqs
    # unweighted entropy in each column, shape: (n_columns,)
    # entropy = 1/log(n_residual_types)
    entropy_scales = np.log(n_residual_types)
    entropy_scales[n_residual_types == 1] = 1
    entropy_scales = np.log(len(alphabet))
    frequencies_smoothed = (occurrences + 1).astype(np.float64)/(n_seqs + len(alphabet))
    entropies = 1 + np.sum(frequencies_smoothed*np.log(frequencies_smoothed), axis=1)/entropy_scales

    stats = {'name': name,
        'n_seqs': n_seqs, 
        'n_columns': n_columns, 
        'conservation_avg': np.mean(conservation),
        'mutation_count_avg': np.mean(n_mutations),
        'gap_fraction_avg': np.mean(gaps),
        'entropy_avg': np.mean(entropies)}

    logger.info('create statistics file: ' + args.output_file)
    with open(args.output_file, 'w') as f:
        for key, val in stats.items():
            f.write('%s\t%s\n'%(key, val))

def plot_hmm_align_stats(args):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    sns.set()

    # plot distribution of statistics for each family
    def plot_stats_boxplot(stat_name, ylabel=None):
        n_figures = 5
        fig, axes = plt.subplots(n_figures, 1, figsize=(14, n_figures*3))
        sns.boxplot(x='repClass', y=stat_name, data=hmmalign_stats, 
                    fliersize=3, color='w', linewidth=1.2, ax=axes[0])
        sns.boxplot(x='repFamily', y=stat_name, data=hmmalign_stats,
                    fliersize=3, color='w', linewidth=1.2, ax=axes[1])
        data = hmmalign_stats.query('(repFamily == "Alu") and (n_seqs >= 10)')
        sns.barplot(x='repName', y=stat_name, data=data,
                    edgecolor='gray', color='w', linewidth=1.2, ax=axes[2])
        data = hmmalign_stats.query('(repFamily == "snRNA") and (n_seqs >= 10)')
        sns.barplot(x='repName', y=stat_name, data=data,
                    edgecolor='gray', color='w', linewidth=1.2, ax=axes[3])
        data = hmmalign_stats.query('(repFamily == "MIR") and (n_seqs >= 10)')
        sns.barplot(x='repName', y=stat_name, data=data,
                    edgecolor='gray', color='w', linewidth=1.2, ax=axes[4])

        for i in range(n_figures):
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
            axes[i].set_ylabel(ylabel)
        plt.tight_layout()
        return axes

    logger.info('read repeat class file: ' + args.repclass_file)
    rep_table = pd.read_table(args.repclass_file,
                          header=None, names=['repName', 'repClass', 'repFamily'])
    logger.info('read hmmalign stats file: ' + args.input_file)
    hmmalign_stats = pd.read_table(args.input_file)
    hmmalign_stats.rename(columns={'name': 'repName'}, inplace=True)
    hmmalign_stats.index = hmmalign_stats['repName']
    hmmalign_stats = pd.merge(hmmalign_stats, rep_table, on='repName')

    logger.info('create output file: ' + args.output_file)
    with PdfPages(args.output_file) as pdf:
        # number of sequences for each repeat family
        axes = plot_stats_boxplot('n_seqs', ylabel='Number of sequences')
        pdf.savefig()
        plt.close()
        # average number of mutations
        axes = plot_stats_boxplot('mutation_count_avg', ylabel='Average number of mutations')
        axes[0].set_ylim(0, 300)
        axes[1].set_ylim(0, 300)
        pdf.savefig()
        plt.close()
        # fraction of gaps
        axes = plot_stats_boxplot('gap_fraction_avg', ylabel='Fraction of gaps')
        pdf.savefig()
        plt.close()
        # average conservation
        axes = plot_stats_boxplot('conservation_avg', ylabel='Average conservation')
        pdf.savefig()
        plt.close()

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Analyze TE structures')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('get_te_icshape',
                                   help='get TE icSHAPE data given TE regins in transcriptomic coordinates')
    parser.add_argument('--bed-file', type=str, required=True,
                        help='bed file that specify transcriptomic intervals of TE regions')
    parser.add_argument('--icshape-file', type=str, required=True,
                        help='all icSHAPE data in GenomicData format')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output file in GenomicData format')
    parser.add_argument('--feature', type=str, default='icshape',
                        help='feature name in the icSHAPE file')
    
    parser = subparsers.add_parser('hmm_align_stats',
                                help='read output of hmmalign and output some statistics')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='Stockholm format')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='prefix for output files')

    parser = subparsers.add_parser('plot_alignment',
                                help='read output of hmmalign and output some statistics')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='Stockholm format')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='prefix for output files')

    parser = subparsers.add_parser('plot_hmm_align_stats')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='output file of hmm_align_stats')
    parser.add_argument('--repclass-file', type=str, required=True,
                        help='a text file containing 3 columns: repName, repClass, repFamily')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output plot PDF file')
    
    parser = subparsers.add_parser('hmm_align_reactivities',
                        help='map reactivities to HMM alignment results')
    parser.add_argument('--alignment-dir', '-a', type=str, required=True,
                        help='directory containing alignments in Stockholm format (.sto)')
    parser.add_argument('--reactivity-file', '-r', type=str, required=True,
                        help='reactivities in GenomicData format')
    parser.add_argument('--feature', type=str, required=True,
                        help='feature use in the reactivity file')
    parser.add_argument('--min-records', type=int, default=5,
                        help='minimal number of records needed to be kept')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output HDF5 file')

    parser = subparsers.add_parser('plot_hmm_align_reactivities',
                        help='plot results of hmm_align_reactivities')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='output file hmm_align_reactivities')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output plot PDF file')

    args = main_parser.parse_args()
    logger = logging.getLogger('te_structure.' + args.command)

    if args.command == 'get_te_icshape':
        get_te_icshape(args)
    elif args.command == 'hmm_align_stats':
        hmm_align_stats(args)
    elif args.command == 'plot_alignment':
        plot_alignment(args)
    elif args.command == 'plot_hmm_align_stats':
        plot_hmm_align_stats(args)
    elif args.command == 'hmm_align_reactivities':
        hmm_align_reactivities(args)
    elif args.command == 'plot_hmm_align_reactivities':
        plot_hmm_align_reactivities(args)