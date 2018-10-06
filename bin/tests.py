#! /usr/bin/env python
from __future__ import print_function
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')

command_handlers = {}
def command_handler(f):
    command_handlers[f.__name__] = f
    return f

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Tests')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('iterative_enrich')
    parser.add_argument('--input-file', '-i', type=str, required=True,
        help='FASTA file containing motifs')
    parser.add_argument('--alphabet', '-a', type=str, default='AUCG')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max-iter', type=int, default=200,
        help='maximum number of iteration')
    parser.add_argument('--model-file', type=str,
        help='output file prefix for models')
    parser.add_argument('--pred-file', type=str,
        help='output file (HDF5 format) for predictions')

    args = main_parser.parse_args()
    if args.command is None:
        raise ValueError('empty command')
    logger = logging.getLogger('tests.' + args.command)

    import numpy as np
    command_handlers.get(args.command)(args)