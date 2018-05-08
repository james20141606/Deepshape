#! /usr/bin/env python
from __future__ import print_function
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')


def get_icshape(args):
    import numpy as np
    import h5py
    import pandas as pd
    from tqdm import tqdm
    from genomic_data import GenomicData

    logger.info('read bed file: ' + args.bed_file)
    bed = pd.read_table(args.bed_file, header=None)

    logger.info('read icSHAPE data file: ' + args.icshape_file)
    icshape = {}
    with h5py.File(args.icshape_file, 'r') as fin:
        for key in fin.keys():
            icshape[key] = fin[key][:]

    logger.info('create output file: ' + args.output_file)
    fout = h5py.File(args.output_file, 'w')
    for row in tqdm(bed.itertuples(index=False), total=bed.shape[0]):
        transcript_id = row[0]
        peak_id =str(row[3])
        data = icshape.get(transcript_id)
        if data is not None:
            fout.create_dataset('%s,%s,%d,%d'%(peak_id, transcript_id, row[1], row[2]), data=data[row[1]:row[2]])
    fout.close()

def create_dataset(args):
    import numpy as np
    import h5py
    import pandas as pd
    from tqdm import tqdm
    from formats import read_fasta

    logger.info('read peak file: ' + args.peak_file)
    peaks = pd.read_table(args.peak_file,
        names=['chrom', 'start', 'end', 'peak_id', 'label', 'strand'])
    peaks['peak_id'] = peaks['peak_id'].astype('U')
    peaks.index = peaks['peak_id']
    logger.info('read sequence file: ' + args.sequence_file)
    sequences = {name:seq for name, seq in read_fasta(args.sequence_file)}
    logger.info('read reactivity file: ' + args.reactivity_file)
    reactivities = {}
    with h5py.File(args.reactivity_file, 'r') as f:
        for peak_id in f.keys():
            reactivities[peak_id.split(',')[0]] = f[peak_id][:]
    peak_ids = list(sorted(reactivities.keys()))

    def onehot_encode(x, alphabet='ATCG'):
        alphabet = np.frombuffer(bytearray(alphabet, encoding='ascii'), dtype='S1')
        x_shape = list(x.shape)
        encoded = (x.reshape(x_shape + [1]) == alphabet.reshape([1]*len(x_shape) + [-1])).astype(np.int32)
        return encoded

    X_seq = np.concatenate([np.frombuffer(bytearray(sequences[peak_id], encoding='ascii'), dtype='S1')[np.newaxis, :] for peak_id in peak_ids], axis=0)
    X_seq = onehot_encode(X_seq)
    X_r   = np.concatenate([reactivities[peak_id][np.newaxis, :, np.newaxis] for peak_id in peak_ids], axis=0)
    # imputate reactivities with median values
    X_r[np.isnan(X_r)] = np.nanmedian(X_r.flatten())
    X = np.concatenate([X_seq, X_r], axis=2)
    y = peaks['label'][peak_ids]
    logger.info('create output file: ' + args.output_file)
    with h5py.File(args.output_file, 'w') as fout:
        fout.create_dataset('X', data=X)
        fout.create_dataset('y', data=y)

def summarize_metrics(args):
    import numpy as np
    import h5py
    import pandas as pd
    from tqdm import tqdm
    from ioutils import open_file_or_stdout

    def parse_filename(filename):
        c = filename.split('/')
        d = {'dataset': c[2],
            'cv_index': c[4].split('_')[-1],
            'model_name': c[-1].split('.')[1],
            'icshape_dataset': c[-1].split('.')[2]
        }
        return d

    summary = []
    for input_file in args.input_files:
        d = parse_filename(input_file)
        with h5py.File(input_file, 'r') as f:
            d['accuracy'] = f['metrics/accuracy'][()]
            d['roc_auc'] = f['metrics/roc_auc'][()]
            summary.append(d)
    summary = pd.DataFrame.from_records(summary)
    summary = summary[['dataset', 'icshape_dataset', 'model_name', 'cv_index', 'accuracy', 'roc_auc']]
    with open_file_or_stdout(args.output_file) as fout:
        summary.to_csv(fout, sep='\t', index=False)

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Analyze TE structures')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('get_icshape',
                                   help='get icSHAPE data in given intervals in transcriptomic coordinates')
    parser.add_argument('--bed-file', type=str, required=True,
                        help='bed file that specify transcriptomic intervals')
    parser.add_argument('--icshape-file', type=str, required=True,
                        help='all icSHAPE data in GenomicData format')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output file in GenomicData format')
    parser.add_argument('--feature', type=str, default='icshape',
                        help='feature name in the icSHAPE file')

    parser = subparsers.add_parser('create_dataset')
    parser.add_argument('--peak-file', type=str, required=True, 
        help='peaks.bed')
    parser.add_argument('--sequence-file', type=str, required=True, 
        help='peaks.transcript_coord.extended.fa')
    parser.add_argument('--reactivity-file', type=str, required=True, 
        help='peaks.icshape.Lu_2016_invitro')
    parser.add_argument('--output-file', '-o', type=str, required=True,
        help='HDF5 file containing X and y')
    
    parser = subparsers.add_parser('summarize_metrics')
    parser.add_argument('--input-files', '-i', type=str, required=True, nargs='+',
        help='input directory containing all metrics')
    parser.add_argument('--output-file', '-o', type=str, default='-',
        help='a text summary table file')
    
    args = main_parser.parse_args()
    if args.command is None:
        raise ValueError('empty command')
    logger = logging.getLogger('find_motif.' + args.command)

    import numpy as np

    if args.command == 'get_icshape':
        get_icshape(args)
    elif args.command == 'create_dataset':
        create_dataset(args)
    elif args.command == 'summarize_metrics':
        summarize_metrics(args)
