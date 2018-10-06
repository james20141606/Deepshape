#! /usr/bin/env python
from __future__ import print_function
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')

command_handlers = {}
def command_handler(f):
    command_handlers[f.__name__] = f
    return f

def set_keras_num_threads(n_threads):
    from keras import backend as K
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = n_threads
    config.inter_op_parallelism_threads = n_threads
    K.set_session(tf.Session(config=config))

def make_gaussian_mixture(n_samples=100, n_features=2, n_classes=2):
    mu = np.random.normal(scale=2, size=(1, n_classes, n_features))
    sigma = np.random.uniform(0.2, 0.6, size=(1, n_classes, n_features))
    X0 = np.random.normal(size=(n_samples, n_classes, n_features))
    X = X0*sigma + mu
    labels = np.random.choice(n_classes, size=n_samples)
    return X[np.r_[:n_samples], labels, :], labels

@command_handler
def vae(args):
    from generative_models import SimpleEncoder, SimpleDecoder, VariationalAutoencoder

    n_latent = args.n_latent
    input_shape = (args.n_features,)
    vae = VariationalAutoencoder(input_shape=input_shape,
        encoder=SimpleEncoder(n_latent),
        decoder=SimpleDecoder(input_shape))
    vae.model.summary()
    X, labels = make_gaussian_mixture(n_samples=5000, n_features=2, n_classes=10)
    vae.fit(X, epochs=args.epochs)

@command_handler
def vae_seq(args):
    from vae import MLPEncoder, SequenceDecoder, VariationalAutoencoder
    from utils import set_keras_num_threads, sequences_to_windows, fasta_to_onehot

    logger.info('read input sequences: ' + args.input_file)
    X = fasta_to_onehot(args.input_file, alphabet=args.alphabet)
    input_shape = X.shape[1:]
    vae = VariationalAutoencoder(input_shape=input_shape,
        encoder=MLPEncoder(args.n_latent, layers=[32]),
        decoder=SequenceDecoder(input_shape),
        loss='sequence_vae_loss',
        n_sampler=args.n_sampler)
    vae.model.summary()
    vae.fit(X, epochs=args.epochs, batch_size=args.batch_size)
    metric_values = vae.model.evaluate(X, X, batch_size=args.batch_size)
    if args.model_file is not None:
        logger.info('save model weights to file: ' + args.model_file)
        vae.model.save_weights(args.model_file)
    if args.metrics_file is not None:
        logger.info('save metrics to file: ' + args.metrics_file)
        with open(args.metrics_file, 'w') as fout:
            for metric_name, metric_value in zip(vae.model.metrics_names, metric_values):
                fout.write('{}\t{}\n'.format(metric_name, metric_value))

@command_handler
def mixture_vae_seq(args):
    import h5py
    from generative_models import BackgroundModel
    from vae import MLPEncoder, SequenceDecoder, VariationalAutoencoder
    from utils import set_keras_num_threads, sequences_to_windows, windows_to_sequences, fasta_to_onehot
    from mixture import MixtureModel
    from tensorboardX import SummaryWriter

    logger.info('read input sequence file: ' + args.input_file)
    X_seq = fasta_to_onehot(args.input_file, alphabet=args.alphabet)
    if args.max_n_seq is not None:
        X_seq = X_seq[:args.max_n_seq]
    
    discover_length = args.length
    logger.info('split sequences to windows')
    X_win, starts, ends = sequences_to_windows(X_seq, discover_length)
    input_shape = (discover_length, len(args.alphabet))
    vae = VariationalAutoencoder(input_shape=input_shape,
        encoder=MLPEncoder(args.n_latent, layers=[32]),
        decoder=SequenceDecoder(input_shape),
        likelihood='sequence',
        n_sampler=args.n_sampler,
        batch_size=100,
        epochs=args.epochs)
    set_keras_num_threads(2)
    mixture = MixtureModel(models=[BackgroundModel(length=discover_length), vae], logger=logger)
    mixture.init_params(components=True)

    logger.info('train the model')
    summary_writer = SummaryWriter(log_dir=args.log_dir)
    mixture.fit(X_win, n_runs=args.n_runs, max_iter=args.max_iter, summary_writer=summary_writer)
    summary_writer.close()

    if args.posteriors_file:
        logger.info('compute posteriors')
        posteriors = windows_to_sequences(mixture.posteriors(X_win), starts, ends)
        logger.info('save posteriors: ' + args.posteriors_file)
        with h5py.File(args.posteriors_file, 'w') as f:
            for i in range(len(posteriors)):
                f.create_dataset(str(i), data=posteriors[i])

    logger.info('save mixture model parameters: ' + args.model_file + '.mixture.h5')
    with h5py.File(args.model_file + '.mixture.h5', 'w') as f:
        mixture.to_hdf5(f)
    logger.info('save VAE model weights: ' + args.model_file + '.weights.h5')
    vae.model.save_weights(args.model_file + '.weights.h5')

@command_handler
def posteriors_to_pwm(args):
    import h5py
    from utils import fasta_to_onehot, pwm_to_transfac, pwm_to_weblogo

    logger.info('read input sequence file: ' + args.sequence_file)
    X_seq = fasta_to_onehot(args.sequence_file, alphabet=args.alphabet)
    
    posterior_threshold = args.min_odds_ratio/(1.0 + args.min_odds_ratio)
    logger.info('read posteriors file: ' + args.posteriors_file)
    locations = []
    with h5py.File(args.posteriors_file, 'r') as f:
        for seq_id in range(len(f)):
            posteriors = f[str(seq_id)][:]
            max_index = np.argmax(posteriors[:, 1])
            max_value = posteriors[max_index, 1]
            if max_value > posterior_threshold:
                locations.append((seq_id, max_index, max_value))
    
    length = args.length
    X_hit = np.zeros((len(locations), length, len(args.alphabet)))
    for i_loc, loc in enumerate(locations):
        seq_index, max_index, max_value = loc
        X_hit[i_loc] = X_seq[seq_index, max_index:(max_index + length)]
    pwm = np.sum(X_hit, axis=0)

    name = os.path.splitext(os.path.basename(args.sequence_file))[0]
    logger.info('create PWM file: ' + args.pwm_file)
    with open(args.pwm_file, 'w') as f:
        transfac = pwm_to_transfac(pwm, name, alphabet=args.alphabet)
        f.write(transfac)
    
    logger.info('create WebLogo file: ' + args.weblogo_file)
    with open(args.weblogo_file, 'wb') as f:
        output_format = os.path.splitext(os.path.basename(args.weblogo_file))[1][1:]
        f.write(pwm_to_weblogo(pwm, name, alphabet=args.alphabet, 
            output_format=output_format))
    

@command_handler
def mixture_seq(args):
    import h5py
    from generative_models import PwmModel, BackgroundModel
    from mixture import MixtureModel
    from utils import fasta_to_onehot, pwm_to_weblogo, sequences_to_windows, windows_to_sequences

    logger.info('read input sequence file: ' + args.input_file)
    X_seq = fasta_to_onehot(args.input_file, alphabet=args.alphabet)
    if args.max_n_seq is not None:
        X_seq = X_seq[:args.max_n_seq]

    discover_length = args.length
    X_win, starts, ends = sequences_to_windows(X_seq, discover_length)
    mixture = MixtureModel(models=[BackgroundModel(length=discover_length),
        PwmModel(length=discover_length)], logger=logger)
    mixture.init_params(components=True)
    #print('initial parameters: ', mixture.get_params())

    logger.info('train the model')
    mixture.fit(X_win, n_runs=args.n_runs, max_iter=args.max_iter)

    #print('optimized parameters: ', mixture.get_params())
    if args.posteriors_file:
        logger.info('compute posteriors')
        posteriors = windows_to_sequences(mixture.posteriors(X_win), starts, ends)
        logger.info('save posteriors: ' + args.posteriors_file)
        with h5py.File(args.posteriors_file, 'w') as f:
            for i in range(len(posteriors)):
                f.create_dataset(str(i), data=posteriors[i])

    logger.info('save the motif: ' + args.output_file)
    pwm = mixture.get_params()['model[1]']['pwm']
    with open(args.output_file, 'w') as f:
        f.write('\t'.join(args.alphabet) + '\n')
        for i in range(pwm.shape[0]):
            f.write('\t'.join('{:.6f}'.format(a) for a in pwm[i]) + '\n')
        f.close()

    logger.info('save the motif as WebLogo: ' + args.weblogo_file)
    with open(args.weblogo_file, 'wb') as f:
        weblogo = pwm_to_weblogo(pwm, 'PWM', output_format='pdf')
        f.write(weblogo)

@command_handler
def compare_weblogo(args):
    from wand.image import Image
    from wand.drawing import Drawing
    from wand.color import Color
    import math

    if len(args.input_file) !=  len(args.annotation):
        raise ValueError('different number of input files and annotations')
    input_images = []
    for input_file in args.input_file:
        logger.info('read input image:' + input_file)
        input_images.append(Image(filename=input_file, resolution=args.resolution))
    n_cols = min(args.n_cols, len(args.input_file))
    n_rows = int(math.ceil(float(len(input_images))/n_cols))
    input_width = max(img.width for img in input_images)
    input_height = max(img.height for img in input_images)
    logger.info('composite images')
    output_image = Image(width=input_width*n_cols, height=(input_height + args.annotation_size)*n_rows)
    for n in range(len(input_images)):
        i, j = n//n_cols, n%n_cols
        with Drawing() as draw:
            left, top = input_width*j, (input_height + args.annotation_size)*i
            draw.composite(operator='copy', left=left, top=top + args.annotation_size,
                width=input_images[n].width, height=input_images[n].height, image=input_images[n])
            draw.font = 'Arial'
            draw.font_size = args.annotation_size
            draw.font_style = 'normal'
            draw.font_weight = 700
            draw.font_stretch = 'normal'
            draw.text(left + round(input_width/2), args.annotation_size + top, args.annotation[n])
            draw(output_image)
    logger.info('save output image: ' + args.output_file)
    output_image.format = os.path.splitext(args.output_file)[1][1:]
    output_image.save(filename=args.output_file)

@command_handler
def sample_pwm(args):
    from utils import read_transfac
    from utils import sample_pwm as _sample_pwm
    from ioutils import open_file_or_stdout

    logger.info('read motif file: ' + args.input_file)
    motif = read_transfac(args.input_file)
    pwm = motif['PWM']/np.sum(motif['PWM'], axis=1, keepdims=True)
    alphabet = motif['PO'].replace('T', 'U')
    sequences = _sample_pwm(pwm, alphabet=motif['PO'], size=args.n)
    pwm_name = motif['ID']

    logger.info('create output file: ' + args.output_file)
    fout = open_file_or_stdout(args.output_file)
    for i, seq in enumerate(sequences):
        fout.write('>{}_{:06d}\n'.format(pwm_name, i + 1))
        fout.write(seq)
        fout.write('\n')
    fout.close()

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='DeepShape main program')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('vae')
    parser.add_argument('--n-latent', type=int, default=2)
    parser.add_argument('--n-features', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)

    parser = subparsers.add_parser('vae_seq')
    parser.add_argument('--input-file', '-i', type=str, required=True,
        help='input sequences in FASTA format')
    parser.add_argument('--alphabet', '-a', type=str, default='AUCG')
    parser.add_argument('--n-latent', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n-sampler', type=int, default=10,
        help='number of samples to draw from each latent distribution')
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--model-file', type=str,
        help='output file for model weights')
    parser.add_argument('--metrics-file', type=str,
        help='output file for training metrics')

    parser = subparsers.add_parser('mixture_seq',
        help='train mixture model (PWM) on sequences')
    parser.add_argument('--input-file', '-i', type=str, required=True,
        help='FASTA file containing motifs')
    parser.add_argument('--alphabet', '-a', type=str, default='AUCG')
    parser.add_argument('--length', '-l', type=int, default=8,
        help='length of the motif to discover')
    parser.add_argument('--n-runs', type=int, default=20,
        help='number of runs for the EM algorithm')
    parser.add_argument('--max-iter', type=int, default=200,
        help='maximum number of iteration for each EM run')
    parser.add_argument('--max-n-seq', type=int, default=None,
        help='maximum number of sequences to use for training')
    parser.add_argument('--weblogo-file', type=str)
    parser.add_argument('--output-file', '-o', type=str, required=True,
        help='output file for the discovered motif')
    parser.add_argument('--posteriors-file', type=str,
        help='output file (HDF5 format) for posterior probabilites')

    parser = subparsers.add_parser('mixture_vae_seq',
        help='train mixture model (VAE) on sequences')
    parser.add_argument('--input-file', '-i', type=str, required=True,
        help='FASTA file containing motifs')
    parser.add_argument('--alphabet', '-a', type=str, default='AUCG')
    parser.add_argument('--length', '-l', type=int, default=8,
        help='length of the motif to discover')
    parser.add_argument('--n-latent', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n-sampler', type=int, default=10,
        help='number of samples to draw from each latent distribution')
    parser.add_argument('--n-runs', type=int, default=20,
        help='number of runs for the EM algorithm')
    parser.add_argument('--max-iter', type=int, default=200,
        help='maximum number of iteration for each EM run')
    parser.add_argument('--max-n-seq', type=int, default=1000,
        help='maximum number of sequences to use for training')
    parser.add_argument('--log-dir', type=str, 
        help='output directory for Tensorboard')
    parser.add_argument('--model-file', type=str,
        help='output file prefix for models')
    parser.add_argument('--posteriors-file', type=str,
        help='output file (HDF5 format) for posterior probabilites')
    
    parser = subparsers.add_parser('compare_weblogo',
        help='combine multiple weblogo images to a single image')
    parser.add_argument('--input-file', '-i', type=str, action='append', required=True,
        help='input WebLogo image files')
    parser.add_argument('--annotation', '-a', type=str, action='append', required=True,
        help='annotation text for each image file')
    parser.add_argument('--resolution', '-r', type=int, default=200,
        help='resolution for reading input images')
    parser.add_argument('--annotation-size', '-s', type=int, default=20,
        help='font size for annnotation text')
    parser.add_argument('--output-file', '-o', type=str, required=True,
        help='output image')
    parser.add_argument('--n-cols', type=int, default=4)

    parser = subparsers.add_parser('sample_pwm',
        help='sample sequences from a PWM')
    parser.add_argument('--input-file', '-i', type=str, required=True,
        help='input motif file in transfac format')
    parser.add_argument('-n', type=int, default=10,
        help='number of sequence to sample')
    parser.add_argument('--output-file', '-o', type=str, default='-',
        help='output file in FASTA format')
    
    parser = subparsers.add_parser('posteriors_to_pwm',
        help='convert discovered motif instances to PWM')
    parser.add_argument('--sequence-file', '-s', type=str, required=True)
    parser.add_argument('--posteriors-file', '-p', type=str, required=True)
    parser.add_argument('--length', '-l', type=int, required=True)
    parser.add_argument('--alphabet', '-a', type=str, default='AUCG')
    parser.add_argument('--min-odds-ratio', type=float, default=1.0)
    parser.add_argument('--pwm-file', type=str, required=True)
    parser.add_argument('--weblogo-file', type=str, required=True)

    args = main_parser.parse_args()
    if args.command is None:
        raise ValueError('empty command')
    logger = logging.getLogger('deepshape.' + args.command)

    import numpy as np
    command_handlers.get(args.command)(args)
