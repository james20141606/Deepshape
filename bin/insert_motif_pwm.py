#! /usr/bin/env python
import sys, argparse

def autofile(name, mode, buffering=None):
	if isinstance(name, file):
		return name
	elif name == '-':
		if 'w' in mode:
			return sys.stdout
		else:
			return sys.stdin
	else:
		return open(name, mode)

def is_realfile(name):
	if isinstance(name, file):
		return False
	if name != '-':
		return False
	return True

def _defered_import():
	import numpy as np
	import scipy.stats
	globals().update(locals())

def random_pwm(length, alphabet='ATCG', min_info=0.2, max_info=1.0):
	"""returns an array of shape (length, len(alphabet))
	"""
	pwm = np.zeros((length, len(alphabet)), dtype='float')
	norm_factor = np.log(len(alphabet))
	for i in range(length):
		info = 0
		while (info < min_info) or (info > max_info):
			x = np.random.rand(len(alphabet))
			info = 1.0 - scipy.stats.entropy(x)/norm_factor
		pwm[i, :] = x/x.sum()
	return pwm

def save_pwm(filename, pwm, alphabet):
	f = autofile(filename, 'w')
	f.write('\t'.join(alphabet))
	f.write('\n')
	np.savetxt(f, pwm)
	if is_realfile(filename):
		f.close()

def load_pwm(filename):
	f = autofile(filename, 'r')
	alphabet = ''.join(f.readline().strip().split())
	pwm = np.loadtxt(f)
	if is_realfile(filename):
		f.close()
	return (pwm, alphabet)

def ibatch(n, batch_size, return_index=False):
	for i in xrange(n / batch_size):
		yield batch_size if not return_index else (i*batch_size, batch_size)
	remain = n % batch_size
	if remain > 0:
		yield remain if not return_index else (n - remain, remain)
	
def sample_pwm(n_seqs, pwm, alphabet, batch_size=100):
	alphabet = np.frombuffer(alphabet, dtype='S1')
	seqs = []
	for n_batch in ibatch(n_seqs, batch_size):
		seqs_batch = np.zeros((n_batch, pwm.shape[0]), dtype='S1')
		for i in range(pwm.shape[0]):
			indices = np.random.choice(len(alphabet), size=n_batch, p=pwm[i, :])
			seqs_batch[:, i] = alphabet[indices]
		for i in range(n_batch):
			seqs.append(np.getbuffer(seqs_batch[i]))
	return seqs

def sample_pwm_generator(pwm, alphabet, batch_size=100):
	alphabet = np.frombuffer(alphabet, dtype='S1')
	seqs = []
	while True:
		seqs_batch = np.zeros((batch_size, pwm.shape[0]), dtype='S1')
		for i in range(pwm.shape[0]):
			indices = np.random.choice(len(alphabet), size=batch_size, p=pwm[i, :])
			seqs_batch[:, i] = alphabet[indices]
		for i in range(batch_size):
			yield np.getbuffer(seqs_batch[i])

def insert_pwm(infile, outfile, pwm, alphabet):
	fin = autofile(infile, 'r')
	fout = autofile(outfile, 'w')
	generator = sample_pwm_generator(pwm, alphabet)
	for line in fin:
		if line.startswith('>'):
			fout.write(line)
		else:
			seq = bytearray(line.strip())
			motif = next(generator)
			pos = np.random.randint(len(seq) - len(motif))
			seq[pos:(pos + len(motif))] = motif
			fout.write(str(seq))
			fout.write('\n')
	if is_realfile(infile):
		fin.close()
	if is_realfile(outfile):
		fout.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	subparsers = parser.add_subparsers(title='command', help='sub-command help')

	parser_generate = subparsers.add_parser('generate', help='generate a random PWM')
	parser_generate.set_defaults(command='generate')
	parser_generate.add_argument('-a', '--alphabet', type=str, default='ATCG', required=False)
	parser_generate.add_argument('-l', '--length', type=int, default=10, required=False,
		help='motif length')
	parser_generate.add_argument('--min-info', type=float, default=0.3, required=False,
		help='minimum information content')
	parser_generate.add_argument('--max-info', type=float, default=1.0, required=False,
		help='maximum information content')
	parser_generate.add_argument('-o', '--outfile', type=str, default='-', required=False,
		help='output motif file')

	parser_sample = subparsers.add_parser('sample', help='sample sequences from a PWM')
	parser_sample.set_defaults(command='sample')
	parser_sample.add_argument('-n', '--nseqs', type=int, default=10, required=False,
		help='number of sequences to sample')
	parser_sample.add_argument('-i', '--infile', type=str, default='-', required=False,
		help='input PWM file')
	parser_sample.add_argument('-o', '--outfile', type=str, default='-', required=False,
		help='output sequences file')

	parser_insert = subparsers.add_parser('insert', help='insert motifs into a FASTA file')
	parser_insert.set_defaults(command='insert')
	parser_insert.add_argument('-i', '--infile', type=str, default='-', required=False,
		help='input FASTA file')
	parser_insert.add_argument('-o', '--outfile', type=str, default='-', required=False,
		help='output FASTQ file')
	parser_insert.add_argument('-p', '--pwmfile', type=str, required=True,
		help='input PWM file')

	args = parser.parse_args()

	_defered_import()

	if args.command == 'generate':
		pwm = random_pwm(args.length, args.alphabet, min_info=args.min_info, max_info=args.max_info)
		save_pwm(args.outfile, pwm, args.alphabet)
	elif args.command == 'sample':
		pwm, alphabet = load_pwm(args.infile)
		seqs = sample_pwm(args.nseqs, pwm, alphabet)
		f = autofile(args.outfile, 'w')
		for seq in seqs:
			f.write(seq)
			f.write('\n')
		if is_realfile(args.outfile):
			f.close()
	elif args.command == 'insert':
		pwm, alphabet = load_pwm(args.pwmfile)
		insert_pwm(args.infile, args.outfile, pwm, alphabet)


