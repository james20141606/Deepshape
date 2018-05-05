#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')

def calc_dms_reactivities(treatment, control):
    """
    Reference: Ding, Y., Tang, Y., Kwok, C.K., Zhang, Y., Bevilacqua, P.C., and Assmann, S.M. (2014).
        In vivo genome-wide profiling of RNA secondary structure reveals novel regulatory features. Nature 505, 696-700.
    """
    assert treatment.shape == control.shape
    N = treatment.shape[0]
    nan_mask = (np.isnan(treatment) | np.isnan(control))
    treatment = np.log(np.nan_to_num(treatment) + 1)
    control = np.log(np.nan_to_num(control) + 1)
    treatment_mean = treatment.mean()
    if treatment_mean <= 0:
        return np.full(N, np.nan)
    control_mean = control.mean()
    if control_mean <= 0:
        return np.full(N, np.nan)
    P = treatment/treatment_mean
    M = control/control_mean
    scores = P - M
    scores[scores < 0] = 0
    pct2 = np.percentile(scores, 98)
    pct8 = np.percentile(scores, 92)
    scale = scores[np.logical_and(pct8 <= scores, scores <= pct2)].mean()
    if np.isclose(scale, 0.0):
        scores[:] = np.nan
    else:
        scores /= scale
        scores[scores > 7.0] = 7.0
    scores[nan_mask] = np.nan
    return scores

def dms_seq(args):
    import numpy as np
    import pandas as pd
    import h5py
    from bx.bbi.bigwig_file import BigWigFile
    from tqdm import tqdm

    bigwig_treatment = {}
    bigwig_control = {}
    logger.info('read treatment plus file: ' + args.treatment_plus_file)
    bigwig_treatment['+'] = BigWigFile(open(args.treatment_plus_file, 'rb'))
    logger.info('read treatment minus file: ' + args.treatment_minus_file)
    bigwig_treatment['-'] = BigWigFile(open(args.treatment_minus_file, 'rb'))
    logger.info('read control plus file: ' + args.control_plus_file)
    bigwig_control['+'] = BigWigFile(open(args.control_plus_file, 'rb'))
    logger.info('read control minus file: ' + args.control_minus_file)
    bigwig_control['-'] = BigWigFile(open(args.control_minus_file, 'rb'))
    logger.info('read bed file: ' + args.bed_file)
    bed = pd.read_table(args.bed_file, header=None)
    logger.info('create output file: ' + args.output_file)
    fout = h5py.File(args.output_file, 'w')

    for c in tqdm(bed.itertuples(index=False), total=bed.shape[0], unit='transcript'):
        n_exons = int(c[9])
        exon_sizes  = np.asarray([int(a) for a in c[10].split(',')[:n_exons]])
        exon_starts = np.asarray([int(a) for a in c[11].split(',')[:n_exons]])
        exon_ends = exon_starts + exon_sizes
        treatment = bigwig_treatment[c[5]].get_as_array(c[0], c[1], c[2])
        if treatment is None:
            continue
        treatment = np.concatenate([treatment[start:end] for start, end in zip(exon_starts, exon_ends)])
        control = bigwig_control[c[5]].get_as_array(c[0], c[1], c[2])
        if control is None:
            continue
        control = np.concatenate([control[start:end] for start, end in zip(exon_starts, exon_ends)])
        reactivities = calc_dms_reactivities(treatment, control)
        # reverse the array on minus strand
        if c[5] == '-':
            reactivities = reactivities[::-1]
        if not np.all(np.isnan(reactivities)):
            fout.create_dataset(c[3], data=reactivities.astype(np.float32))
    fout.close()


if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Calculate reactivities for structure profiling data')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('dms_seq')
    parser.add_argument('--treatment-plus-file', type=str, required=True,
                        help='RT-stop counts of treatment sample in BigWig format')
    parser.add_argument('--treatment-minus-file', type=str, required=True,
                        help='RT-stop counts of treatment sample in BigWig format')
    parser.add_argument('--control-plus-file', type=str, required=True,
                        help='RT-stop counts of control sample in BigWig format')
    parser.add_argument('--control-minus-file', type=str, required=True,
                        help='RT-stop counts of control sample in BigWig format')
    parser.add_argument('--bed-file', type=str, required=True,
                        help='transcript annotations in BED12 format')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output reactivities in HDF5 format')

    args = main_parser.parse_args()
    logger = logging.getLogger('calculate_reactivity.' + args.command)

    import numpy as np

    if args.command == 'dms_seq':
        dms_seq(args)

