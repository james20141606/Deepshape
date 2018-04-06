#! /usr/bin/env python
# Extract information from the output files produced by the icSHAPE pipeline:
#   https://github.com/qczhang/icSHAPE

import argparse, sys, os
from common import FastaReader

def icshape_to_hdf5(infile, outfile):
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
    import numpy as np
    name = []
    length = []
    rpkm = []
    icshape = []
    with open(infile, 'r') as fin:
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

    length = np.asarray(length, dtype='int64')
    rpkm = np.asarray(rpkm, dtype='float64')
    icshape_length = np.asarray(map(len, icshape), dtype='int64')
    start = np.cumsum(icshape_length) - icshape_length

    fout = h5py.File(outfile, 'w')
    fout.create_dataset('n_seqs', data=np.asarray(len(name), dtype='int64'))
    fout.create_dataset('name', data=np.asarray(name, dtype='S'))
    fout.create_dataset('length', data=length)
    fout.create_dataset('start', data=start)
    fout.create_dataset('end', data=start + icshape_length)
    fout.create_dataset('rpkm', data=rpkm)
    fout.create_dataset('icshape', data=np.concatenate(icshape))
    fout.close()

def add_extra_lines(fin, n=1):
    for line in fin:
        yield line
    for i in range(n):
        yield ''

def rt_to_hdf5(infile, outfile, normalized=False):
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
    import numpy as np
    name = []
    length = []
    rpkm = []
    bd = []
    rt = []
    if normalized:
        base_frequency_bd = []
        base_frequency_rt = []
    with open(infile, 'r') as fin:
        n_lines = 0
        prev_name = None
        prev_fields = []
        for line in add_extra_lines(fin):
            if line.startswith('#'):
                continue
            n_lines += 1
            fields = line.strip().split('\t')
            if normalized:
                if (len(fields) == 0) or ((fields[0] != prev_name) and (prev_name is not None)):
                    print prev_fields[0][0]
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

    length = np.asarray(length, dtype='int64')
    rpkm = np.asarray(rpkm, dtype='float64')
    rt_length = np.asarray(map(len, rt), dtype='int64')
    start = np.cumsum(rt_length) - rt_length

    fout = h5py.File(outfile, 'w')
    fout.create_dataset('n_seqs', data=np.asarray(len(name), dtype='int64'))
    fout.create_dataset('name', data=np.asarray(name, dtype='S'))
    fout.create_dataset('length', data=length)
    fout.create_dataset('start', data=start)
    fout.create_dataset('end', data=start + rt_length)
    fout.create_dataset('rpkm', data=rpkm)
    fout.create_dataset('base_density', data=np.concatenate(bd))
    fout.create_dataset('rt_stop', data=np.concatenate(rt))
    if normalized:
        fout.create_dataset('base_frequency_bd', data=np.asarray(base_frequency_bd, dtype='float64'))
        fout.create_dataset('base_frequency_rt', data=np.asarray(base_frequency_rt, dtype='float64'))
    fout.close()

def query(infile, names, fields):
    fin = h5py.File(infile, 'r')
    name_to_index = {}
    for i, name in enumerate(fin['name'][:]):
        name_to_index[name.tostring()] = i
    start = fin['start'][:]
    end = fin['end'][:]
    for name in names:
        i = name_to_index[name]
        columns = [name]
        for field in fields:
            if field in ('length', 'rpkm'):
                columns.append(str(fin[field][i]))
            elif field in ('rt_stop', 'base_density', 'icshape'):
                columns.append(','.join(map(str, fin[field][start[i]:end[i]])))
        print '\t'.join(columns)
    fin.close()

def normalize_rt(infile):
    fin = h5py.File(infile, 'r')
    start = fin['start'][:]
    end = fin['end'][:]
    name = fin['name'][:]
    rt_stop = fin['rt_stop'][:]
    fin.close()
    for i in range(len(name)):
        print '{}\t{}'.format(name[i], rt_stop[start[i]:end[i]].mean())

def icshape_to_text(icshape_file, sequence_file, outfile, rpkm_pct=0.0):
    """
    Arguments:
        icshape_file: HDF5 file generated by icshape_to_hdf5()
        sequence_file: FASTA file in which sequence names match the 'name' field in the HDF5 file
        outfile: a text file by concatenation of multiple sections.
            Each section begins with a line of two columns: (sequence name, sequence length).
            The remaining lines of a section contains 3 columns: (1-based index, base, icSHAPE score)
    """
    import numpy as np
    sequences = {}
    for name, seq in FastaReader(sequence_file):
        sequences[name] = seq
    f_icshape = h5py.File(icshape_file, 'r')
    seqnames = f_icshape['name'][:]
    length = f_icshape['length'][:]
    start = f_icshape['start'][:]
    end = f_icshape['end'][:]
    rpkm = f_icshape['rpkm'][:]
    icshape = f_icshape['icshape'][:]
    f_icshape.close()
    rpkm_cutoff = np.percentile(rpkm, rpkm_pct)
    fout = open(outfile, 'w')
    for i in range(len(seqnames)):
        if rpkm[i] < rpkm_cutoff:
            continue
        fout.write('{}\t{}\n'.format(seqnames[i], length[i]))
        values = icshape[start[i]:end[i]]
        seq = sequences[seqnames[i]]
        for j in range(length[i]):
            icshape_str = '{:.3f}'.format(values[j]) if not np.isnan(values[j]) else 'NULL'
            fout.write('{}\t{}\t{}\n'.format(j + 1, seq[j], icshape_str))
    fout.close()

def read_icshape(filename):
    # seqname => (length, rpkm, values)
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            seqname = fields[0].split('.')[0]
            values = np.asarray([float(v) if v != 'NULL' else np.nan for v in fields[3:]],
                dtype='float32')
            data[seqname] = (int(fields[1]), float(fields[2]), values)
    return data

def icshape_correlate(file1, file2):
    """Calculate correlation coefficient between two icSHAPE files
    The comparison is at transcript level.
    """
    from scipy.stats import pearsonr
    data1 = read_icshape(file1)
    data2 = read_icshape(file2)
    seqnames = list(set(data1.keys()) & set(data2.keys()))
    for seqname in seqnames:
        values1 = data1[seqname][2]
        values2 = data2[seqname][2]
        ind = np.nonzero(np.logical_not(np.isnan(values1)) & np.logical_not(np.isnan(values2)))
        if len(ind) > 0:
            r, pvalue = pearsonr(values1[ind], values2[ind])
            print '{}\t{}\t{}\t{}\t{}'.format(seqname, data1[seqname][1], data2[seqname][1], r, pvalue)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract information from outputs of the icSHAPE pipeline')
    parser.add_argument('command', type=str,
        choices=('rt_to_hdf5', 'normalized_rt_to_hdf5', 'query', 'icshape_to_hdf5', 'normalize_rt', 'icshape_to_text',
            'icshape_correlate'))
    parser.add_argument('-i', '--infile', type=str, required=False,
        help='input file name')
    parser.add_argument('--infile2', type=str, required=False,
        help='second input file name')
    parser.add_argument('-o', '--outfile', type=str, required=False,
        help='output file name')
    parser.add_argument('-s', '--seqfile', type=str, required=False,
        help='input sequence file name')
    parser.add_argument('--names', type=str, required=False, default='',
        help='a comma-separated list of sequence names to query')
    parser.add_argument('--fields', type=str, required=False, default='',
        help='a comma-separated list of fields names to query')
    parser.add_argument('--rpkm-pct', type=float, required=False, default=0,
        help='only transcripts with rpkm percentile higher than this value is kept')
    args = parser.parse_args()

    import h5py
    import numpy as np
    if args.command == 'rt_to_hdf5':
        rt_to_hdf5(args.infile, args.outfile)
    if args.command == 'normalized_rt_to_hdf5':
        rt_to_hdf5(args.infile, args.outfile, normalized=True)
    elif args.command == 'query':
        query(args.infile, args.names.split(','), args.fields.split(','))
    elif args.command == 'icshape_to_hdf5':
        icshape_to_hdf5(args.infile, args.outfile)
    elif args.command == 'normalize_rt':
        normalize_rt(args.infile)
    elif args.command == 'icshape_to_text':
        icshape_to_text(args.infile, args.seqfile, args.outfile,
            rpkm_pct=args.rpkm_pct)
    elif args.command == 'icshape_correlate':
        icshape_correlate(args.infile, args.infile2)
