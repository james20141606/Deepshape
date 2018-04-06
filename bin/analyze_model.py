#! /usr/bin/env python
from common import CommandLineTool, Argument, make_dir
import os, sys
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
import logging
logging.basicConfig(level=logging.DEBUG)

class AnalyzeLogRegWeights(CommandLineTool):
    description = 'Plot Logistic regression weights in a Keras saved model file'
    arguments = [Argument('model_file', type=str, required=True, help='a Keras model file'),
        Argument('outfile', type=str, required=True, help='output plot file'),
        Argument('alphabet', type=str, default='ATCG')]
    def __call__(self):
        model_weights = h5py.File(self.model_file, 'r')['/model_weights/dense_1/dense_1_W:0'][:]
        window_size = model_weights.shape[0]/len(self.alphabet)
        model_weights = model_weights.reshape((window_size, 4))
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 9), sharey=True)
        for i in range(len(self.alphabet)):
            ax = axes[i]
            ax.bar(np.arange(window_size), model_weights[:, i], color='b', edgecolor='none')
            ax.set_xticks(np.arange(window_size, step=5))
            ax.set_xlim(0, window_size)
            ax.set_title('Logistic regression weights ({})'.format(self.alphabet[i]))
            ax.set_ylabel('Weight')
        self.logger.info('savefig: {}'.format(self.outfile))
        make_dir(os.path.dirname(self.outfile))
        plt.tight_layout()
        plt.savefig(self.outfile, dpi=150, bbox_inches='tight')

class ReportMetrics(CommandLineTool):
    description = 'Generate a summary report of prediction metrics'
    arguments = [Argument('infiles', short_opt='-i', type=str, required=True, nargs='+',
        help='HDF5 file with metrics stored in the /metrics group')]
    def __call__(self):
        for infile in self.infiles:
            fin = h5py.File(infile, 'r')
            grp = fin['metrics']
            model_name = os.path.splitext(os.path.basename(infile))[0]
            for metric in fin['metrics'].keys():
                print '{}\t{}\t{}'.format(model_name, metric, grp[metric][()])
            fin.close()

if __name__ == '__main__':
    CommandLineTool.from_argv()()
