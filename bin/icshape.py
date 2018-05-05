#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
import numpy as np

def icshape_to_hdf5(args):
    import numpy as np
    import h5py
    from tqdm import tqdm

    logger.info('read input file: ' + args.input_file)
    fin = open(args.input_file, 'r')
    logger.info('create output file: ' + args.output_file)
    fout = h5py.File(args.output_file, 'w')
    for line in tqdm(fin, unit='transcript'):
        c = line.strip().split('\t')
        reactivities = np.asarray(c[3:])
        reactivities[reactivities == 'NULL'] = 'nan'
        reactivities = reactivities.astype(np.float32)
        fout.create_dataset(c[0], data=reactivities, compression=True)
    fout.close()
    fin.close()

def rt_to_hdf5(args):
    import numpy as np
    import h5py
    from tqdm import tqdm
    from itertools import groupby

    logger.info('read input file: ' + args.input_file)
    fin = open(args.input_file, 'r')
    logger.info('create output RT-stop counts file: ' + args.rt_file)
    f_rt = h5py.File(args.rt_file, 'w')
    logger.info('create output base density file: ' + args.bd_file)
    f_bd = h5py.File(args.bd_file, 'w')
    for transcript_id, group in tqdm(groupby(map(lambda x: x.strip().split('\t'), fin), key=lambda x: x[0]), unit='transcript'):
        for i, c in enumerate(group):
            counts = np.asarray(c[3:])
            counts[counts == 'NULL'] = 'nan'
            counts = counts.astype(np.float32)
            if i == 0:
                f_rt.create_dataset(transcript_id, data=counts, compression=True)
            elif i == 1:
                f_bd.create_dataset(transcript_id, data=counts, compression=True)
    f_bd.close()
    f_rt.close()
    fin.close()

def flagstat(args):
    import pysam
    from ioutils import open_file_or_stdin, open_file_or_stdout

    logger.info('read input file: ' + args.input_file)
    fin = open_file_or_stdin(args.input_file)
    sam = pysam.AlignmentFile(fin, 'rb')
    counts = [0]*4096
    for read in sam:
        counts[read.flag] += 1
    sam.close()

    logger.info('create output file: ' + args.output_file)
    with open_file_or_stdout(args.output_file) as fout:
        fout.write('flag\tcounts\n')
        for flag, count in enumerate(counts):
            if count > 0:
                fout.write('{}\t{}\n'.format(flag, count))

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Process SHAPE-MaP data')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('icshape_to_hdf5')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='icshape.out file')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output file in HDF5 format')
    
    parser = subparsers.add_parser('rt_to_hdf5')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='icshape.out file')
    parser.add_argument('--rt-file', '-r', type=str, required=True,
                        help='output file containing RT-stop counts in HDF5 format')
    parser.add_argument('--bd-file', '-b', type=str, required=True,
                        help='output file containing base densities in HDF5 format')
    
    parser = subparsers.add_parser('flagstat')
    parser.add_argument('--input-file', '-i', type=str, default='-',
                        help='input BAM/SAM file')
    parser.add_argument('--output-file', '-o', type=str, default='-',
                        help='output file for stats')

    args = main_parser.parse_args()
    if args.command is None:
        raise ValueError('command is empty')
    logger = logging.getLogger('icshape.' + args.command)

    if args.command == 'icshape_to_hdf5':
        icshape_to_hdf5(args)
    elif args.command == 'rt_to_hdf5':
        rt_to_hdf5(args)
    elif args.command == 'flagstat':
        flagstat(args)