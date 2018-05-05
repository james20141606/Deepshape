#! /usr/bin/env python
from __future__ import print_function
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')

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
    
    args = main_parser.parse_args()
    logger = logging.getLogger('find_motif.' + args.command)

    if args.command == 'get_icshape':
        get_icshape(args)
