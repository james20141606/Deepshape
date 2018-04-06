import sys, os, re
from collections import namedtuple
from ioutils import append_extra_line

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

class FastaReader(object):
    def __init__(self, filename):
        self.f = RollBackFile(open(filename, 'r'))
        self.filename = filename
        self.lineno = 0

    def close(self):
        self.f.close()

    def get_record(self):
        line = self.f.readline()
        self.lineno += 1
        if len(line) == 0:
            raise StopIteration()
        while not line.startswith('>'):
            line = self.f.readline()
        name = line.strip()[1:]
        seq = []
        line = self.f.readline()
        self.lineno += 1
        while (len(line) > 0) and (not line.startswith('>')):
            seq.append(line.strip())
            line = self.f.readline()
            self.lineno += 1
        if line.startswith('>'):
            self.f.unreadline()
        if len(seq) == 0:
            rint >>sys.stderr, 'Warning: empty record in {} at line {}'.format(self.filename, self.lineno)
            return None
        else:
            seq = ''.join(seq)
            return (name, seq)


    def __iter__(self):
        return self

    def next(self):
        record = self.get_record()
        while not record:
            record = self.get_record()
        return record

def read_hdf5(filename):
    import h5py
    f = h5py.File(filename, 'r')
    data = {}
    for key in f.keys():
        if f[key].shape == ():
            data[key] = f[key][()]
        else:
            data[key] = f[key][:]
    f.close()
    return data

class IndexedFastaReader(object):
    def __init__(self, filename, index_file=None):
        self.filename = filename
        self.index_file = filename + '.fai' if index_file is None else index_file
        self.read_index()
        self.fasta_f = open(self.filename, 'rb')

    def read_index(self):
        """Fasta index format spec: http://www.htslib.org/doc/faidx.html
        5 columns: name, length, offset, line_bases, line_width
        """
        self.index = {}
        with open(self.index_file, 'r') as f:
            for lineno, line in enumerate(f):
                c = line.strip().split()
                if len(c) != 5:
                    raise ValueError('expects 5 columns in the fasta index file but got {} at line '.format(len(c), lineno + 1))
                self.index[c[0]] = map(int, c[1:])

    def get(self, name, offset=0, length=None):
        if name not in self.index:
            return None
        else:
            ind = self.index[name]
            n_lines = ind[0]/ind[2]
            seq = ''
            self.fasta_f.seek(ind[1])
            for i in xrange(ind[0]/ind[2]):
                seq += self.fasta_f.read(ind[3])[:-1]
            remaining = ind[0] % ind[2]
            if remaining != 0:
                seq += self.fasta_f.read(remaining)
            return seq

    def __getitem__(self, name):
        return self.get(name)

    def __iter__(self):
        for name in self.names:
            yield self.get(name)

    def close(self):
        self.fasta_f.close()

Bed12Record = namedtuple('Bed12Record',
    ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand',
     'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts'])
def read_bed12(bedfile):
    with open(bedfile, 'r') as f:
        for lineno, line in enumerate(f):
            fields = line.strip().split()
            if len(fields) != 12:
                raise ValueError('expect 12 columns but got only {} columns at line {}'.format(len(fields), lineno + 1))
            fields[1] = int(fields[1])
            fields[2] = int(fields[2])
            fields[9] = int(fields[9])
            fields[10] = [int(a) for a in fields[10].split(',') if a != '']
            fields[11] = [int(a) for a in fields[11].split(',') if a != '']
            yield Bed12Record._make(fields)

def read_rnafold(filename, parse_energy=True):
    """RNAfold output files is an extended FASTA format
    An extra line is added after each sequence: <structure> (energy).
    For example:
    >sequence
    CACCCCAUAGGGC
    ...(((....))) (-8.5)
    """
    import re
    if parse_energy:
        pat_line2 = re.compile(r'([\(.\)]+)\s+\(([ 0-9.-]+)\)')
    else:
        pat_line2 = re.compile(r'([\(.\)]+)')
    with open(filename, 'r') as f:
        name = None
        line = f.readline()
        while line != '':
            if line.startswith('>'):
                if name is not None:
                    yield (name, seq, structure, energy)
                name = line.strip()[1:].split()[0]
                line = f.readline()
                seq = line.strip()
                line = f.readline()
                m = pat_line2.search(line)
                if m:
                    structure = m.group(1)
                    if parse_energy:
                        energy = float(m.group(2).strip())
                    else:
                        energy = 0.0
                else:
                    raise ValueError('invalid structure for sequence {}'.format(name))
            line = f.readline()
        if name is not None:
            yield (name, seq, structure, energy)

def read_ct(filename, num=0):
    """CT format specification: http://rna.urmc.rochester.edu/Text/File_Formats.html
    A CT (Connectivity Table) file contains secondary structure information for a sequence.
    These files are saved with a CT extension.
    When entering a structure to calculate the free energy, the following format must be followed.

    Start of first line: number of bases in the sequence
    End of first line: title of the structure
    Each of the following lines provides information about a given base in the sequence. Each base has its own line, with these elements in order:
    Base number: index n
    Base (A, C, G, T, U, X)
    Index n-1
    Index n+1
    Number of the base to which n is paired. No pairing is indicated by 0 (zero).
    Natural numbering. RNAstructure ignores the actual value given in natural numbering, so it is easiest to repeat n here.
    """
    records = []
    with open(filename, 'r') as f:
        line = f.readline()
        while len(line) > 0:
            fields = line.strip().split()
            length = int(fields[0])
            title = fields[-1].strip()
            seq = bytearray(length)
            pairs = [0] * length

            for j in range(length):
                fields = f.readline().strip().split()
                i = int(fields[0])
                seq[i - 1] = fields[1]
                pair = int(fields[4])
                if pair > 0:
                    pairs[i - 1] = pair
                else:
                    pairs[i - 1] = 0
            records.append((title, str(seq), pairs))
            line = f.readline()
    if num < 0:
        return records
    else:
        return records[num]

def structure_to_pairs(structure):
    """Return a list of positions to which each base is paired to
    For example: '..(((..)))' => [0, 0, 9, 8, 7, 0, 0, 4, 3, 2]
    Arguments:
        structure: a dot-bracket representation of RNA structure
    """
    pairs = [0]*len(structure)
    S = []
    for i in range(len(structure)):
        if structure[i] == '.':
            pairs[i] = 0
        elif structure[i] in '([':
            S.append(i)
        elif structure[i] in ')]':
            if len(S) == 0:
                raise ValueError('invalid structure')
            j = S.pop()
            pairs[i] = j + 1
            pairs[j] = i + 1
        else:
            raise ValueError('invalid character %s found in the structure'%structure[i])
    return pairs

def make_pair_list(pairs):
    """Returns a list of pairs (1-based index) in tuples
    For example: [0, 0, 10, 9, 8, 0, 0, 5, 4, 3] => [(2, 9), (3, 8), (4, 7)]
    Arguments:
        pairs: same as the last column in a CT file.
    """
    pair_list = []
    for i in range(len(pairs)):
        if (pairs[i] > 0) and ((i + 1) < pairs[i]):
            pair_list.append((i + 1, pairs[i]))
    return pair_list

def score_structure(true_pairs, pred_pairs, exact=False):
    """Same as scorer in the RNAstructure package: http://rna.urmc.rochester.edu/Text/scorer.html
    pred_pairs, true_pairs: list of pairs (tuples)
    """
    hash_pair = lambda x: x[0]*1000000 + x[1]
    # make hash for pairs
    pred_hash = set(map(hash_pair, pred_pairs))
    true_hash = set(map(hash_pair, true_pairs))
    scores = {}
    scores['true_pairs'] = len(true_hash)
    scores['pred_pairs'] = len(pred_hash)
    if exact:
        tp = len(pred_hash & true_hash)
        tp_in_true = tp
        tp_in_pred = tp
    else:
        tp_in_pred = 0
        for i, j in pred_pairs:
            if any(map(lambda x: hash_pair(x) in true_hash,
                [(i, j), (i, j - 1), (i, j + 1), (i + 1, j), (i - 1, j)])):
                tp_in_pred += 1
        tp_in_true = 0
        for i, j in true_pairs:
            if any(map(lambda x: hash_pair(x) in pred_hash,
                [(i, j), (i, j - 1), (i, j + 1), (i + 1, j), (i - 1, j)])):
                tp_in_true += 1
    scores['tp_in_true'] = tp_in_true
    scores['tp_in_pred'] = tp_in_pred
    scores['sensitivity'] = float(tp_in_true)/len(true_hash) if len(true_hash) > 0 else 0
    scores['ppv'] = float(tp_in_pred)/len(pred_hash) if len(pred_hash) > 0 else 0

    return scores

def read_rme(filename):
    """Read RME input format
    First line is a header: (name, 1-based position, pred, base)
    Return a nested dict with values indexed as: value[name][position]
    Note: returned position is 0-based
    """
    values = {}
    with open(filename, 'r') as f:
        for lineno, line in enumerate(f):
            if lineno == 0:
                continue
            fields = line.strip().split('\t')
            if fields[0] not in values:
                values[fields[0]] = {}
            values[fields[0]][int(fields[1]) - 1] = float(fields[2])
    return values

def read_probing(filename):
    """Read sequences and reactivities from a file.
    The format is similar to FASTA format except that there is an extra line
    after each sequence which is a comma-separated list of values of equal length
    to the sequence.
    For example:
      >sequence
      AUGUACGUAC
      0.01,0.00,0.54,0.23,0.00,1.23,1.02,0.34,0.07,0.91
    Returns a tuple: (sequence name, sequence, values)
    """
    with open(filename, 'r') as f:
        name = None
        line = f.readline()
        while line != '':
            if line.startswith('>'):
                if name is not None:
                    yield (name, seq, structure, energy)
                name = line.strip()[1:].split()[0]
                line = f.readline()
                seq = line.strip()
                line = f.readline()
                values = line.strip(',')
            line = f.readline()
        if name is not None:
            yield (name, seq, structure, energy)

def read_hdf5_dataset(filename, return_name=False):
    """
    Read a dataset from an HDF5 file
    :param filename: file path and dataset name separated by ":" (e.g file.h5:dataset)
    :return: the dataset
    """
    import h5py
    if ':' not in filename:
        raise ValueError('missing dataset name in the HDF5 file: ' + filename)
    i = filename.index(':')
    f = h5py.File(filename[:i], 'r')
    dataset = filename[(i + 1):]
    data = f[dataset][:]
    f.close()
    if return_name:
        return data, dataset
    else:
        return data