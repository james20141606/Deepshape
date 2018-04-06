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

    args = main_parser.parse_args()
    logger = logging.getLogger('te_structure.' + args.command)

    if args.command == 'get_te_icshape':
        get_te_icshape(args)