#! /usr/bin/env python
from cmdtool import CommandLineTool, Argument
from ioutils import prepare_output_file
import sys, os

def get_model(model_name, hyperparam={}):
    if model == 'svm':
        from sklearn.svm import SVC
        model = SVC(**hyperparam)
    elif model == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**hyperparam)
    return model

def open_file_or_stdin(filename):
    if filename == '-':
        return sys.stdin
    else:
        return open(filename, 'r')

class TrainClassifier(CommandLineTool):
    arguments = [Argument('train_file', short_opt='-i', type=str, required=True),
        Argument('cv_index_file', type=str, help='CV index created by CreateCvIndex'),
        Argument('cv_fold', type=int),
        Argument('model_name', type=str, required=True, choices=('SVM', 'RandomForest'), help='name of the classifier'),
        Argument('model_file', short_opt='-o', type=str, required=True,
            help='file path for saving the model (in Python pickle format)'),
        Argument('valid_metric_file', type=str),
        Argument('hyperparam', type=str, help='model hyper-parameter in JSON format')
        ]

    def __call__(self):
        import json
        import h5py
        import cPickle
        from sklearn.metrics import roc_auc_score, accuracy_score

        hyperparam = json.loads(self.hyperparam)
        model = get_model(self.model_name, hyperparam)

        self.logger.info('load data: {}'.format(self.infile))
        fin = h5py.File(self.infile, 'r')
        X_train = fin['X_train'][:]
        y_train = fin['y_train'][:]
        fin.close()
        X_valid = None
        y_valid = None

        if self.cv_index_file is not None:
            if self.cv_fold is None:
                raise ValueError('argument --cv-fold is required if --cv-index-file is specified')
            self.logger.info('load CV index: ' + self.cv_index_file)
            f = h5py.File(self.cv_index_file, 'r')
            train_index = f[str(self.cv_fold)]['train'][:]
            test_index = f[str(self.cv_fold)]['test'][:]
            f.close()
            X_valid = X_train[test_index]
            y_valid = y_train[test_index]
            X_train = X_train[train_index]
            y_train = y_train[train_index]

        self.logger.info('train the model')
        model.fit(X_train, y_train)
        self.logger.info('save model: {}'.format(self.outfile))
        prepare_output_file(self.outfile)
        with open(self.outfile, 'w') as f:
            cPickle.dump(model, self.outfile)

        if X_valid:
            self.logger.info('validate the model')
            y_pred_labels = model.predict(X_valid)
            self.logger.info('save the metrics: ' + self.valid_metric_file)
            prepare_output_file(self.valid_metric_file)
            f = h5py.File(self.valid_metric_file, 'w')
            f.create_dataset('model_name', dtype='S', data=self.model_name)
            f.create_dataset('hyperparam', dtype='S', data=json.dumps(self.hyperparam))
            f.create_dataset('y_pred_labels', data=y_pred_labels)
            f.create_dataset('y_true', data=y_true)
            g = f.create_group('metrics')
            g.create_dataset('accuracy', data=accuracy_score(y_valid, y_pred_labels))
            f.close()

class CreateCvIndex(CommandLineTool):
    arguments = [Argument('n_samples', short_opt='-n', type=int, required=True,
            help='number of samples'),
        Argument('n_folds', short_opt='-k', type=int, required=True,
            help='number of folds'),
        Argument('outfile', short_opt='-o', type=str, required=True,
            help='output file in HDF5 format with dataset names: /<fold>/train, /<fold>/test')]
    def __call__(self):
        import h5py
        from sklearn.model_selection import KFold
        import numpy as np

        self.logger.info('save file: ' + self.outfile)
        prepare_output_file(self.outfile)
        fout = h5py.File(self.outfile, 'w')
        kfold = KFold(self.n_folds, shuffle=True)
        fold = 0
        for train_index, test_index in kfold.split(np.arange(self.n_samples)):
            g = fout.create_group('%d'%fold)
            g.create_dataset('train', data=train_index)
            g.create_dataset('test', data=test_index)
            fold += 1
        fout.close()

class HyperParamGrid(CommandLineTool):
    description = 'Expand a grid specification of hyperparameters into a list in JSON format'
    arguments = [Argument('infile', short_opt='-i', default='-',
            help='parameter ranges in JSON format'),
        Argument('sample', type=int,
            help='randomly sample up to a certain number of parameters')]
    def __call__(self):
        import json
        import itertools
        import random
        fin = open_file_or_stdin(self.infile)
        grid_spec = json.load(fin)
        fin.close()

        fout = sys.stdout
        param_names = []
        param_values = []
        for name, value in grid_spec.iteritems():
            param_names.append(name)
            param_values.append(value)
        param_list = []
        for param in itertools.product(*param_values):
            param_list.append(dict(zip(param_names, param)))
        if self.sample:
            param_list = random.sample(param_list, min(len(param_list), self.sample))
        for param in param_list:
            fout.write(json.dumps(param) + '\n')

class SelectBestModel(CommandLineTool):
    arguments = [Argument('metric_dir', short_opt='-i', type=str, required=True,
            help='directory containing prediction metric files in HDF5 format with datasets: y_true, y_pred'),
        Argument('metric', type=str, default='accuracy'),
        Argument('outfile', short_opt='-o', type=str, required=True)]
    def __call__(self):
        import h5py


if __name__ == '__main__':
    CommandLineTool.from_argv()()
