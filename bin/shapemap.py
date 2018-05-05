#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
import numpy as np

def shape_to_hdf5(args):
    import numpy as np
    import pandas as pd
    import h5py
    from tqdm import tqdm

    logger.info('read input directory: ' + args.input_dir)
    logger.info('create output file: ' + args.output_file)
    fout = h5py.File(args.output_file, 'w')
    fout.create_group('sequences')
    fout.create_group('reactivities')
    logger.info('use file extension: ' + args.extension)
    input_files = [os.path.join(args.input_dir, filename) for filename in os.listdir(args.input_dir) if filename.endswith(args.extension)]
    for input_file in tqdm(input_files, unit='file'):
        tx_id = os.path.splitext(os.path.basename(input_file))[0]
        df = pd.read_table(input_file, sep='\s+',
            header=None, names=['index', 'reactivity', 'stderr', 'nucleotide'])
        fout.create_dataset('reactivities/' + tx_id, data=df['reactivity'].values.astype(np.float32))
        fout.create_dataset('stderr/' + tx_id, data=df['stderr'].values.astype(np.float32))
        fout.create_dataset('sequences/' + tx_id, data=''.join(df['nucleotide'].values))
    fout.close()

def get_orf_reactivities(args):
    import pandas as pd
    import numpy as np
    import h5py

    logger.info('read input file: ' + args.input_file)
    with h5py.File(args.input_file, 'r') as f:
        sequences = {name:f['sequences/' + name][()] for name in f['sequences'].keys()}
        reactivities = {name:f['reactivities/' + name][:] for name in f['reactivities'].keys()}
    
    logger.info('read ORF file: ' + args.input_file)
    orfs_trancript_coord = pd.read_table(args.orf_file, 
                                     header=None, names=['chrom', 'start', 'end', 'name', 'score', 'strand'])
    orfs_trancript_coord.index = orfs_trancript_coord['name']
    orfs_trancript_coord['chrom'] = orfs_trancript_coord['chrom'].astype('S')

    logger.info('create output file: ' + args.output_file)
    fout = h5py.File(args.output_file, 'w')
    fout.create_group('sequences')
    fout.create_group('reactivities')
    for i, row in enumerate(orfs_trancript_coord.itertuples(index=False)):
        x = reactivities[row[0]][row[1]:row[2]]
        fout.create_dataset('reactivities/' + row[3], data=reactivities[row[0]][row[1]:row[2]])
        fout.create_dataset('sequences/' + row[3], data=sequences[row[0]][row[1]:row[2]])

def shape_to_patterna(args):
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from ioutils import open_file_or_stdout

    logger.info('read input directory: ' + args.input_dir)
    logger.info('use file extension: ' + args.extension)
    input_files = [os.path.join(args.input_dir, filename) for filename in os.listdir(args.input_dir) if filename.endswith(args.extension)]
    logger.info('create output file: ' + args.output_file)
    fout = open_file_or_stdout(args.output_file)
    for input_file in tqdm(input_files, unit='file'):
        df = pd.read_table(input_file, header=None, sep='\s+', na_values=args.na_values,
            names=('index', 'reactivity', 'stderr', 'nucleotide'))
        reactivities = df['reactivity'].values.copy()
        #reactivities[reactivities < -100] = np.nan
        seq_name = os.path.splitext(os.path.basename(input_file))[0]
        fout.write('>' + seq_name + '\n')
        if args.sequence:
            rna_sequence = df['nucleotide'].values.copy()
            rna_sequence[rna_sequence == 'T'] = 'U'
            fout.write(''.join([str(a) for a in rna_sequence]))
        else:
            fout.write(' '.join(['%.4f'%a for a in reactivities]))
        fout.write('\n')
    fout.close()

def patterna_input_to_hdf5(args):
    import h5py
    import numpy as np

    logger.info('read input file: ' + args.input_file)
    logger.info('create output file: ' + args.output_file)
    fout = h5py.File(args.output_file, 'w')
    with open(args.input_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                name = line[1:].strip()
            else:
                values = np.asarray(line.strip().split()).astype(np.float32)
                fout.create_dataset(name, data=values)
    fout.close()
            
def read_patteRNA_posteriors(filename):
    posteriors1 = {}
    posteriors2 = {}
    with open(filename, 'r') as f:
        for lineno, line in enumerate(f):
            if lineno%3 == 0:
                name = line[1:].strip()
            elif lineno%3 == 1:
                posteriors1[name] = np.asarray(line.strip().split(), dtype=np.float32)
            elif lineno%3 == 2:
                posteriors2[name] = np.asarray(line.strip().split(), dtype=np.float32)
    return posteriors1, posteriors2

def read_patteRNA_viterbi(filename):
    viterbi = {}
    with open(filename, 'r') as f:
        for lineno, line in enumerate(f):
            if lineno%2 == 0:
                name = line[1:].strip()
            elif lineno%2 == 1:
                viterbi[name] = np.asarray(list(line.strip()), dtype='S1').astype(np.int32)
    return viterbi

def patterna_to_hdf5(args):
    import numpy as np
    import pandas as pd
    import h5py
    from tqdm import tqdm

    logger.info('read input dir: ' + args.input_dir)
    if args.posteriors_paired:
        logger.info('read posteriors: ' + os.path.join(args.input_dir, 'posteriors.txt'))
        _, posteriors_paired = read_patteRNA_posteriors(os.path.join(args.input_dir, 'posteriors.txt'))
        logger.info('create output file: ' + args.output_file)
        with h5py.File(args.output_file, 'w') as f:
            for name in posteriors_paired:
                f.create_dataset(name, data=posteriors_paired[name])
    elif args.viterbi:
        logger.info('read viterbi: ' + os.path.join(args.input_dir, 'viterbi.txt'))
        viterbi = read_patteRNA_viterbi(os.path.join(args.input_dir, 'viterbi.txt'))
        logger.info('create output file: ' + args.output_file)
        with h5py.File(args.output_file, 'w') as f:
            for name in viterbi:
                f.create_dataset(name, data=viterbi[name])
    else:
        raise ValueError('unknown patteRNA item: ' + args.item)

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Process SHAPE-MaP data')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('shape_to_hdf5')
    parser.add_argument('--input-dir', '-i', type=str, required=True,
                        help='directory containing .shape files')
    parser.add_argument('--extension', '-x', type=str, default='.map',
                        help='extension of input file names')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output file in HDF5 format')

    parser = subparsers.add_parser('get_orf_reactivities')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='input file in HDF5 format produced by shape_to_hdf5')
    parser.add_argument('--orf-file', type=str, required=True,
                        help='BED file of ORF intervals in transcript coordinates')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output file in HDF5 format')
    
    parser = subparsers.add_parser('shape_to_patterna')
    parser.add_argument('--input-dir', '-i', type=str, required=True,
                        help='directory containing .map files')
    parser.add_argument('--extension', '-x', type=str, default='.map',
                        help='extension of input file names')
    parser.add_argument('--sequence', '-s', action='store_true',
                        help='output sequences instead of reactivities')
    parser.add_argument('--na-values', type=str, help='string to represent NaN')
    parser.add_argument('--output-file', '-o', type=str, default='-',
                        help='output file in HDF5 format')

    parser = subparsers.add_parser('patterna_to_hdf5')
    parser.add_argument('--input-dir', '-i', type=str, required=True,
                        help='output directory of patteRNA')
    parser.add_argument('--output-file', '-o', type=str, default='-',
                        help='prefix of output file in HDF5 format')
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--viterbi', action='store_true', 
                        help='read viterbi from patteRNA output files')
    g.add_argument('--posteriors-paired', action='store_true', 
                        help='item to read from the patteRNA output files')
    
    parser = subparsers.add_parser('patterna_input_to_hdf5')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='input file of patteRNA (similar to FASTA format)')
    parser.add_argument('--output-file', '-o', type=str, default='-',
                        help='prefix of output file in HDF5 format')

    args = main_parser.parse_args()
    if args.command is None:
        raise ValueError('command is empty')
    logger = logging.getLogger('shapemap.' + args.command)

    if args.command == 'shape_to_hdf5':
        shape_to_hdf5(args)
    elif args.command == 'get_orf_reactivities':
        get_orf_reactivities(args)
    elif args.command == 'shape_to_patterna':
        shape_to_patterna(args)
    elif args.command == 'patterna_to_hdf5':
        patterna_to_hdf5(args)
    elif args.command == 'patterna_input_to_hdf5':
        patterna_input_to_hdf5(args)
    