#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')

def annotate_structure_diagram(args):
    import re

    pat_nucleotide = re.compile(r'^\([AUGCT]\) ([0-9\.\-]+) ([0-9\.\-]+) lwstring$')
    prolog_lines = []
    nucleotide_lines = []
    epilog_lines = []
    end_prolog = False
    nucleotide_coords = []
    # read input PostScript file
    logger.info('read input file: ' + args.input_file)
    with open(args.input_file, 'r') as fin:
        for line in fin:
            m = pat_nucleotide.match(line.strip())
            if m is not None:
                x, y = [float(a) for a in m.groups()]
                nucleotide_coords.append((x + 3, y + 3))
                end_prolog = True
                nucleotide_lines.append(line)
            else:
                if end_prolog:
                    epilog_lines.append(line)
                else:
                    prolog_lines.append(line)
    colors = []
    if args.colors is not None:
        import numpy as np
        logger.info('read colors from file: ' + args.colors)
        with open(args.colors, 'r') as f:
            for line in f:
                colors.append([float(a) for a in line.strip().split()])
        colors = np.asarray(colors)
    elif args.values is not None:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.pyplot import get_cmap
        logger.info('use matplotlib colormap: ' + args.colormap)
        colormap = get_cmap(args.colormap)
        values = []
        logger.info('read values from file: ' + args.values)
        with open(args.values, 'r') as f:
            for line in f:
                values.append(float(line.strip()))
        values = np.asarray(values)
        # normalize values
        values = (values - np.min(values))/(np.max(values) - np.min(values))
        values = (values - 0.5)*args.scale + 0.5
        colors = colormap(values)[:, :3]
    
    if len(colors) != len(nucleotide_coords):
        raise ValueError('number of values ({0}) and number nucleotides ({1}) does not match'.format(len(colors), len(nucleotide_coords)))

    logger.info('create output file: ' + args.output_file)
    with open(args.output_file, 'w') as fout:
        fout.writelines(prolog_lines)
        fout.write('/lwfcarc {newpath gsave setrgbcolor translate scale /rad exch def /ang1 exch def /ang2 exch def\n')
        fout.write('0.0 0.0 rad ang1 ang2 arc fill grestore} def\n')
        for nucleotide_coord, color in zip(nucleotide_coords, colors):
            x, y = nucleotide_coord
            r, g, b = color
            fout.write('360.00 0.00 4.20 1.00 1.00 {:.2f} {:.2f} {:.3f} {:.3f} {:.3f} lwfcarc\n'.format(x, y, r, g, b))
        fout.writelines(nucleotide_lines)
        fout.writelines(epilog_lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process SHAPE-MaP data')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='input structure diagram in PostScript format '
                        '(downloaded from http://www.rna.icmb.utexas.edu/DAT/3C/Structure/index.php)')
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--colors', type=str, 
            help='a text file containing RGB values for each nucleotide per line')
    g.add_argument('--values', type=str, 
            help='a text file containing continous values for each nucleotide per line')
    parser.add_argument('--colormap', type=str, default='Greys_r',
            help='matplotlib colormap name')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--output-file', '-o', type=str, required=True,
            help='output file in PostScript format')
    args = parser.parse_args()

    logger = logging.getLogger('annotate_structure_diagram')
    annotate_structure_diagram(args)
