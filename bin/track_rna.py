#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
import h5py
import seaborn as sns
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input_file')
args = parser.parse_args()

# we need to track every fragments every nuc's origin(from which rna's which position)
# so we have to record: the ith fragments first nuc is from which rna's which position
#then we can calculate metrics like accuracy:  the aim is we use rna's icSHAPE as label, and y_pred should
# remove nan position according to y_test nan index and then calculate the acc

 dict(read_fasta(self.sequence_file))

def read_fasta(filename):
    with open(filename, 'r') as f:
        name = None
        seq = ''
        for line in append_extra_line(f):
            if line.startswith('>') or (len(line) == 0):
                if (len(seq) > 0) and (name is not None):
                    yield (name, seq)
                if line.startswith('>'):
                    name = line.strip()[1:].split()[0]
                    seq = ''
            else:
                if name is None:
                    raise ValueError('the first line does not start with ">"')
                seq += line.strip()
