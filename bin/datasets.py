#! /usr/bin/env python
from __future__ import print_function
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')

command_handlers = {}
def command_handler(f):
    command_handlers[f.__name__] = f
    return f

@command_handler
def pwm(args):
    from utils import read_transfac
    from utils import sample_pwm, random_sequences, embed_pwm
    from ioutils import open_file_or_stdout

    logger.info('read motif file: ' + args.input_file)
    motif = read_transfac(args.input_file)
    pwm = motif['PWM']/np.sum(motif['PWM'], axis=1, keepdims=True)
    alphabet = motif['PO']
    pwm_name = motif['ID']
    motif_length = motif['PWM'].shape[0]

    if args.length is None:
        args.length = motif_length
    if args.length < motif_length:
        raise ValueError('cannot embed motif of length {} into sequence of length{}'.format(motif_length, args.length))

    n_motif_seqs = args.n - round(args.bg_percent*0.01*args.n)
    n_bg_seqs = round(args.bg_percent*0.01*args.n)
    sequences, starts = embed_pwm(pwm, alphabet=alphabet, size=n_motif_seqs, length=args.length)
    ends = starts + motif_length
    if n_bg_seqs > 0:
        sequences += random_sequences(args.length, alphabet=alphabet, size=n_bg_seqs)
        starts = np.append(starts, np.zeros(n_bg_seqs, dtype=np.int32))
        ends = np.append(ends, np.zeros(n_bg_seqs, dtype=np.int32))
    labels = np.zeros(args.n, dtype=np.int32)
    labels[:n_motif_seqs] = 1

    # shuffle orders
    logger.info('generate {} motif sequences and {} background sequences'.format(n_motif_seqs, n_bg_seqs))
    seq_indices = np.random.permutation(args.n)
    sequences = [sequences[i] for i in seq_indices]
    labels = labels[seq_indices]
    starts = starts[seq_indices]
    ends = ends[seq_indices]

    logger.info('create output file: ' + args.output_file)
    fout = open_file_or_stdout(args.output_file)
    for i, seq in enumerate(sequences):
        fout.write('>{}_{:06d},{},{},{}\n'.format(pwm_name, i + 1, labels[i], starts[i], ends[i]))
        fout.write(seq)
        fout.write('\n')
    fout.close()

@command_handler
def rfam(args):
    import numpy as np
    import subprocess
    from Bio import SeqIO
    from io import StringIO
    import re
    from utils import random_sequences
    from ioutils import open_file_or_stdout

    alphabet = 'AUCG'

    # read CM file
    motif_name = 'RFAM'
    with open(args.input_file, 'r') as f:
        for line in f:
            c = line.strip().split()
            if c[0] == 'NAME':
                motif_name = c[1]
                break

    n_motif_seqs = args.n - round(args.bg_percent*0.01*args.n)
    n_bg_seqs = round(args.bg_percent*0.01*args.n)
    # generate motif sequences
    p = subprocess.Popen(['cmemit', '--nohmmonly', '-e', str(args.length), '-N', str(n_motif_seqs),
        args.input_file], stdout=subprocess.PIPE)
    out, _ = p.communicate()
    sequences = []
    starts = np.zeros(n_motif_seqs, dtype=np.int32)
    ends = np.zeros(n_motif_seqs, dtype=np.int32)
    labels = np.zeros(args.n, dtype=np.int32)
    labels[:n_motif_seqs] = 1
    pat_cmemit = re.compile(r'^[^/]+/([0-9]+)\-([0-9]+)$')
    for i, record in enumerate(SeqIO.parse(StringIO(str(out, encoding='ascii')), 'fasta')):
        start, end = pat_cmemit.match(record.id).groups()
        sequences.append(str(record.seq))
        starts[i] = int(start) + 1
        ends[i] = int(end)
    # generate background sequences
    if n_bg_seqs > 0:
        sequences += random_sequences(args.length, alphabet=alphabet, size=n_bg_seqs)
        starts = np.append(starts, np.zeros(n_bg_seqs, dtype=np.int32))
        ends = np.append(ends, np.zeros(n_bg_seqs, dtype=np.int32))
    
    # shuffle orders
    logger.info('generate {} motif sequences and {} background sequences'.format(n_motif_seqs, n_bg_seqs))
    seq_indices = np.random.permutation(args.n)
    sequences = [sequences[i] for i in seq_indices]
    labels = labels[seq_indices]
    starts = starts[seq_indices]
    ends = ends[seq_indices]

    logger.info('create output file: ' + args.output_file)
    fout = open_file_or_stdout(args.output_file)
    for i, seq in enumerate(sequences):
        fout.write('>{}_{:06d},{},{},{}\n'.format(motif_name, i + 1, labels[i], starts[i], ends[i]))
        fout.write(seq)
        fout.write('\n')
    fout.close()

@command_handler
def background(args):
    from utils import random_sequences
    from ioutils import open_file_or_stdout

    sequences = random_sequences(args.length, alphabet=args.alphabet, size=args.n)
    logger.info('create output file: ' + args.output_file)
    fout = open_file_or_stdout(args.output_file)
    for i, seq in enumerate(sequences):
        fout.write('>RANDOM_{:06d}:0\n'.format(i + 1))
        fout.write(seq)
        fout.write('\n')
    fout.close()

@command_handler
def print_fasta(args):
    from ioutils import open_file_or_stdin
    from Bio import SeqIO

    with open_file_or_stdin(args.input_file) as f:
        for record in SeqIO.parse(f, 'fasta'):
            seq_id, label, start, end = record.id.split(',')
            seq = str(record.seq)
            start = int(start)
            end = int(end)
            print('>{}'.format(record.id))
            if label == '1':
                print('{}\x1B[1;31m{}\x1B[0m{}'.format(seq[:start], seq[start:end], seq[end:]))
            else:
                print(seq)
            

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Generate datasets for evaluation')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('pwm')
    parser.add_argument('--input-file', '-i', type=str, required=True,
        help='input motif file in transfac format')
    parser.add_argument('-n', type=int, default=10,
        help='number of sequence to sample')
    parser.add_argument('--length', '-l', type=int,
        help='length of each sequence')
    parser.add_argument('--bg-percent', type=float, default=0,
        help='proportion of background sequences')
    parser.add_argument('--output-file', '-o', type=str, default='-',
        help='output file in FASTA format')

    parser = subparsers.add_parser('rfam')
    parser.add_argument('--input-file', '-i', type=str, required=True,
        help='input motif file in transfac format')
    parser.add_argument('-n', type=int, default=10,
        help='number of sequence to sample')
    parser.add_argument('--length', '-l', type=int,
        help='length of each sequence')
    parser.add_argument('--bg-percent', type=float, default=0,
        help='proportion of background sequences')
    parser.add_argument('--output-file', '-o', type=str, default='-',
        help='output file in FASTA format')
    
    parser = subparsers.add_parser('background')
    parser.add_argument('-n', type=int, default=10,
        help='number of sequence to sample')
    parser.add_argument('--alphabet', '-a', type=str, default='AUCG')
    parser.add_argument('--output-file', '-o', type=str, default='-',
        help='output file in FASTA format')

    parser = subparsers.add_parser('print_fasta')
    parser.add_argument('--input-file', '-i', type=str, default='-')


    args = main_parser.parse_args()
    if args.command is None:
        raise ValueError('empty command')
    logger = logging.getLogger('datasets.' + args.command)

    import numpy as np
    command_handlers.get(args.command)(args)
