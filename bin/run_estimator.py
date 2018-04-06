#! /usr/bin/env python
from cmdtool import CommandLineTool, Argument
from ioutils import prepare_output_file
import sys, os, subprocess
import multiprocessing
import logging

def _get_session():
    """Modified the original get_session() function to change the ConfigProto variable
    """
    global _SESSION
    if tf.get_default_session() is not None:
        session = tf.get_default_session()
    else:
        if _SESSION is None:
            if not os.environ.get('OMP_NUM_THREADS'):
                config = tf.ConfigProto(allow_soft_placement=True)
            else:
                nb_thread = int(os.environ.get('OMP_NUM_THREADS'))
                config = tf.ConfigProto(intra_op_parallelism_threads=nb_thread,
                              		    allow_soft_placement=True)
            config.gpu_options.per_process_gpu_memory_fraction = 0.8
            config.gpu_options.allow_growth = True
            _SESSION = tf.Session(config=config)
        session = _SESSION
    if not _MANUAL_VAR_INIT:
        _initialize_variables()
    return session

def import_keras():
    """Import the heavy modules after command line parsing to accelerate startup process
    """
    import keras
    from keras.models import Sequential
    from keras.layers import Input
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution1D, Convolution2D
    from keras.layers.pooling import MaxPooling1D, MaxPooling2D
    from keras.regularizers import l2, l1, l1_l2
    from keras.layers.normalization import BatchNormalization
    from keras import backend as K
    import keras.backend.tensorflow_backend
    # control GPU memory usage for TensorFlow backend
    if K.backend() == 'tensorflow':
    	# replace the original get_session() function
    	keras.backend.tensorflow_backend.get_session.func_code = _get_session.func_code
    	import tensorflow as tf

    import numpy as np
    globals().update(locals())

def get_model(model_name, hyperparam={}):
    if model_name == 'svm':
        from sklearn.svm import SVC
        model = SVC(**hyperparam)
    elif model_name == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**hyperparam)
    elif model_name == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**hyperparam)
    elif model_name == 'linear_regression':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(**hyperparam)
    elif model_name == 'svr':
        from sklearn.svm import SVR
        model = SVR(**hyperparam)
    else:
        raise ValueError('unknown model name: ' + model_name)
    return model

def get_scorer(name):
    """Returns a function that accept at least two parameters: y_true, y_pred
    """
    import sklearn.metrics
    # See http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    scorers = {
        'accuracy': sklearn.metrics.accuracy_score,
        'average_precision': sklearn.metrics.average_precision_score,
        'f1': sklearn.metrics.f1_score,
        'precision': sklearn.metrics.precision_score,
        'recall': sklearn.metrics.recall_score,
        'roc_auc': sklearn.metrics.roc_auc_score,
        'sensitivity': sklearn.metrics.recall_score,
        'ppv': sklearn.metrics.precision_score,
        'r2': sklearn.metrics.r2_score,
        'mean_squared_error': sklearn.metrics.mean_squared_error
        }
    return scorers[name]

def open_file_or_stdin(filename):
    if filename == '-':
        return sys.stdin
    else:
        return open(filename, 'r')

def open_file_or_stdout(filename):
    if filename == '-':
        return sys.stdout
    else:
        return open(filename, 'w')

def dict_to_args(d, mapping):
    """Map dict to another dict that can be used as function arguments
    Arguments:
        d: a dict that stores the values
        mapping:
            If mapping is a dict, the names in d are renamed according to mapping.
            If mapping is a list, then the names in mapping will be extract from d.
    Returns:
        A dict. Names that cannot be found in d will be ignored.
    """
    if isinstance(mapping, list):
        args = {}
        for name in mapping:
            if name in d:
                args[name] = d[name]
    elif isinstance(mapping, dict):
        args = {}
        for name in mapping:
            if name in d:
                args[mapping[name]] = d[name]
    return args

class MakeRegression(CommandLineTool):
    arguments = [Argument('n_samples', type=int, short_opt='-n', default=100),
        Argument('n_features', type=int, short_opt='-p', default=100),
        Argument('n_informative', type=int, default=10),
        Argument('noise', type=float, default=0.0),
        Argument('bias', type=float, default=0.0),
        Argument('scale_targets', action='store_true',
            help='scale the target values to zero-mean and unit variance'),
        Argument('test_ratio', type=float, default=0.3),
        Argument('outfile', short_opt='-o', type=str, required=True)]
    def __call__(self):
        import h5py
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X, y = make_regression(self.n_samples, self.n_features,
            n_informative=self.n_informative, bias=self.bias, noise=self.noise)
        if self.scale_targets:
            self.logger.info('scale target values using StandardScaler')
            scaler = StandardScaler()
            y = scaler.fit_transform(y.reshape(-1, 1))
            y = y.reshape((-1,))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_ratio)

        self.logger.info('save file: ' + self.outfile)
        prepare_output_file(self.outfile)
        f = h5py.File(self.outfile, 'w')
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('X_test',  data=X_test)
        f.create_dataset('y_test', data=y_test)
        f.close()

class MakeClassification(CommandLineTool):
    description = 'sklearn.datasets.make_classification'
    arguments = [
        Argument('n_samples', short_opt='-n', type=int, default=100),
        Argument('n_features', short_opt='-p', type=int, default=20),
        Argument('n_informative', type=int, default=2),
        Argument('n_redundant', type=int, default=2),
        Argument('n_repeated', type=int, default=0),
        Argument('n_classes', type=int, default=2),
        Argument('n_clusters_per_class', type=int, default=2),
        Argument('flip_y', type=float, default=0.01),
        Argument('class_sep', type=float, default=1.0),
        Argument('shift', type=float, default=0.0),
        Argument('scale', type=float, default=1.0),
        Argument('test_ratio', type=float, default=0.3),
        Argument('outfile', short_opt='-o', type=str, required=True)
    ]
    def __call__(self):
        import h5py
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(self.n_samples, self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            n_repeated=self.n_repeated,
            n_classes=self.n_classes,
            n_clusters_per_class=self.n_clusters_per_class,
            flip_y=self.flip_y,
            class_sep=self.class_sep,
            shift=self.shift,
            scale=self.scale)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_ratio)

        self.logger.info('save file: ' + self.outfile)
        prepare_output_file(self.outfile)
        f = h5py.File(self.outfile, 'w')
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('X_test',  data=X_test)
        f.create_dataset('y_test', data=y_test)
        f.close()

def execute_command(cmd):
    logger = logging.getLogger('execute_command')
    logger.info('execute shell command: ' + cmd)
    return_code = subprocess.call(cmd, shell=True)
    if return_code != 0:
        logger.error('exit due to non-zero return value (%d) of the shell command'%return_code)
        raise ValueError('exit due to errors in the subprocess')

class CvPipeline(CommandLineTool):
    arguments = [Argument('train_file', type=str, required=True),
        Argument('test_file', type=str, required=True),
        Argument('n_folds', type=int, required=True),
        Argument('outdir', short_opt='-o', type=str),
        Argument('param_grid_file', type=str, required=True),
        Argument('model_file', type=str, help='copy the best model to the file'),
        Argument('n_jobs', type=int, default=1),
        Argument('train_args', type=str, default=''),
        Argument('metrics', type=list, default='accuracy'),
        Argument('flatten', action='store_true', help='flatten the input into 2 dimensions'),
        Argument('select_model_metric', type=str, default='accuracy'),
        Argument('script_path', type=str, default='bin/run_estimator.py'),
        Argument('execute', action='store_true', help='execute the commands sequentially'),
        Argument('outfile', type=str, default='-', help='write the commands to a file'),
        Argument('remove_outdir', action='store_true', help='remove the cv dir after CV')]
    def __call__(self):
        import copy
        import json
        import itertools

        variables = {k:self.__dict__[k] for k in ('train_file', 'test_file',
            'n_folds', 'outdir', 'param_grid_file', 'train_args', 'model_file',
            'script_path', 'metrics', 'select_model_metric')}
        variables['metrics_str'] = ','.join(variables['metrics'])
        if self.flatten:
            variables['flatten'] = '--flatten'
        else:
            variables['flatten'] = ''

        with open(self.param_grid_file, 'r') as f:
            grid_spec = json.load(f)
        hyperparams = []
        param_names = grid_spec.keys()
        for hyperparam in itertools.product(*grid_spec.values()):
            hyperparams.append(dict(zip(param_names, hyperparam)))
        cmd = []
        cmd.append('{script_path} CreateCvIndex -i {train_file} --n-folds {n_folds} -o {outdir}/cv_index.h5'.format(**variables))
        cmd.append('{script_path} HyperParamGrid -i {param_grid_file} -o {outdir}/hyperparam.txt'.format(**variables))
        cmd_pre = cmd

        cmd = []
        for param_index, hyperparam in enumerate(hyperparams):
            hyperparam = json.dumps(hyperparam)
            for cv_fold in range(self.n_folds):
                variables.update(locals())
                cmd.append('''{script_path} TrainEstimator -i {train_file}
--cv-index-file {outdir}/cv_index.h5
--cv-fold {cv_fold} --model-file {outdir}/{param_index}/{cv_fold}.model
--valid-metric-file {outdir}/{param_index}/{cv_fold}.valid_metrics
--metrics {metrics_str}
--hyperparam '{hyperparam}'
{flatten}
{train_args}'''.format(**variables))
        cmd_train = cmd

        cmd = []
        cmd.append('''{script_path} SelectBestModel -i {outdir} --metric {select_model_metric} --prefix {outdir}/select_model'''.format(**variables))
        cmd.append('''{script_path} TrainEstimator -i {train_file}
--model-file {outdir}/best_model
--hyperparam-file {outdir}/select_model.best_hyperparam.json
{flatten}
{train_args}'''.format(**variables))
        if self.model_file:
            variables['model_dir'] = os.path.dirname(self.model_file)
            cmd.append('''[ -d {model_dir} ] || mkdir -p {model_dir}'''.format(**variables))
            cmd.append('''cp {outdir}/best_model {model_file}'''.format(**variables))
        cmd.append('''{script_path} TestEstimator -i {test_file}
--model-file {outdir}/best_model
--metrics {metrics_str}
{flatten}
-o {outdir}/best_model.metrics.h5'''.format(**variables))
        if self.remove_outdir:
            cmd.append('''rm -r '{outdir}' '''.format(**variables))
        cmd_post = cmd

        cmd_pre = map(lambda x: x.replace('\n', ' '), cmd_pre)
        cmd_train = map(lambda x: x.replace('\n', ' '), cmd_train)
        cmd_post = map(lambda x: x.replace('\n', ' '), cmd_post)
        if self.execute:
            map(execute_command, cmd_pre)
            if self.n_jobs <= 1:
                map(execute_command, cmd_train)
            else:
                pool = multiprocessing.Pool(self.n_jobs)
                pool.map(execute_command, cmd_train)
            map(execute_command, cmd_post)
        else:
            fout = open_file_or_stdout(self.outfile)
            for line in cmd_pre + cmd_train + cmd_post:
                fout.write(line + '\n')
            fout.close()

class TrainEstimator(CommandLineTool):
    arguments = [Argument('train_file', short_opt='-i', type=str, required=True,
            help='the dataset in HDF5 format, required datasets: X_train, y_train'),
        Argument('cv_index_file', type=str, help='CV index created by CreateCvIndex'),
        Argument('cv_fold', type=int),
        Argument('model_name', type=str, required=True,
            help='name of the classifier'),
        Argument('model_type', type=str, default='sklearn', choices=('sklearn', 'keras')),
        Argument('model_file', short_opt='-o', type=str,
            help='file path for saving the model (in Python pickle format)'),
        Argument('model_script', type=str,
            help='load a model specification from a Python script (should define the model variable)'),
        Argument('valid_metric_file', type=str),
        Argument('flatten', action='store_true', help='flatten the input dataset before applying the model'),
        Argument('regress', action='store_true', help='train a regression model'),
        Argument('metrics', type=list),
        Argument('scale_targets', action='store_true',
            help='scale the targets values by mean and variance'),
        Argument('hyperparam', type=str, default='{}', help='model hyper-parameter in JSON format'),
        Argument('hyperparam_file', type=str, help='model hyper-parameter in JSON format from file')]

    def __call__(self):
        import json
        import h5py
        import cPickle
        import zipfile

        if self.hyperparam_file:
            with open(self.hyperparam_file, 'r') as f:
                hyperparam = json.load(f)
        else:
            hyperparam = json.loads(self.hyperparam)

        self.logger.info('load data: {}'.format(self.train_file))
        fin = h5py.File(self.train_file, 'r')
        X_train = fin['X_train'][:]
        y_train = fin['y_train'][:]
        fin.close()
        X_valid = None
        y_valid = None

        if self.cv_index_file is not None:
            if self.cv_fold is None:
                raise ValueError('argument --cv-fold is required if --cv-index-file is specified')
            if self.valid_metric_file is None:
                raise ValueError('argument --valid-metric-file is required if --cv-index-file is specified')
            self.logger.info('load CV index: ' + self.cv_index_file)
            f = h5py.File(self.cv_index_file, 'r')
            train_index = f[str(self.cv_fold)]['train'][:]
            test_index = f[str(self.cv_fold)]['test'][:]
            f.close()
            X_valid = X_train[test_index]
            y_valid = y_train[test_index]
            X_train = X_train[train_index]
            y_train = y_train[train_index]

        if self.flatten:
            X_train = X_train.reshape((X_train.shape[0], -1))
            self.logger.info('flatten the training data to dimension: (%d, %d)'%X_train.shape)
            if X_valid is not None:
                X_valid = X_valid.reshape((X_valid.shape[0], -1))
                self.logger.info('flatten the validation data to dimension: (%d, %d)'%X_train.shape)

        if self.scale_targets:
            self.logger.info('scale the target values using StandardScaler')
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape((-1,))
            if y_valid is not None:
                y_valid = scaler.transform(y_valid.reshape(-1, 1)).reshape((-1,))

        if self.model_script:
            self.logger.info('create model from script: ' + self.model_script)
            if self.model_type == 'keras':
                self.logger.info('use the keras model')
                #with open(os.path.join(os.path.dirname(__file__), 'import_keras.py'), 'r') as f:
                #    exec compile(f.read(), 'import_keras.py', 'exec')
                import_keras()
                with open(self.model_script, 'r') as f:
                    exec compile(f.read(), self.model_script, 'exec')
                from keras.optimizers import SGD
                optimizer = SGD()
                if self.regress:
                    loss = 'mean_squared_error'
                    metrics = ['mae']
                else:
                    loss = 'binary_crossentropy'
                    metrics = ['acc']
                model.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=metrics)
                model.summary()
            else:
                with open(self.model_script, 'r') as f:
                    exec compile(f.read(), self.model_script, 'exec')
        else:
            self.logger.info('create model by name: ' + self.model_name)
            model = get_model(self.model_name, hyperparam)
        self.logger.info('train the model')
        if self.model_type == 'keras':
            model.fit(X_train, y_train, batch_size=100, epochs=20)
        else:
            self.logger.info('model parameters: ' + json.dumps(model.get_params()))
            model.fit(X_train, y_train)
        if self.model_file:
            self.logger.info('save model: {}'.format(self.model_file))
            prepare_output_file(self.model_file)
            if self.model_type == 'keras':
                model.save(self.model_file)
                f = h5py.File(self.model_file, 'r+')
                f.create_dataset('hyperparam', data=json.dumps(hyperparam))
                f.close()
            else:
                zipf = zipfile.ZipFile(self.model_file, 'w', zipfile.ZIP_DEFLATED)
                zipf.writestr('model', cPickle.dumps(model))
                zipf.writestr('hyperparam', json.dumps(hyperparam))
                zipf.close()

        if X_valid is not None:
            if self.metrics is None:
                if self.regress:
                    self.metrics = ['mean_squared_error', 'r2']
                else:
                    self.metrics = ['accuracy']
            self.logger.info('validate the model')
            if self.regress:
                y_pred = model.predict(X_valid)
            else:
                y_pred_labels = model.predict(X_valid)
            self.logger.info('save the metrics: ' + self.valid_metric_file)
            prepare_output_file(self.valid_metric_file)
            f = h5py.File(self.valid_metric_file, 'w')
            f.create_dataset('model_name', data=self.model_name)
            f.create_dataset('hyperparam', data=json.dumps(self.hyperparam))
            f.create_dataset('y_true', data=y_valid)
            if self.regress:
                f.create_dataset('y_pred', data=y_pred)
            else:
                f.create_dataset('y_pred_labels', data=y_pred_labels)
            g = f.create_group('metrics')
            for metric in self.metrics:
                scorer = get_scorer(metric)
                if self.regress:
                    score = scorer(y_valid, y_pred)
                else:
                    score = scorer(y_valid, y_pred_labels)
                self.logger.info('calculate metric {}: {}'.format(metric, score))
                g.create_dataset(metric, data=score)
            if self.scale_targets:
                g.create_dataset('scale_y_mean', data=scaler.mean_)
                g.create_dataset('scale_y_std', data=scaler.scale_)
            f.close()

class TestEstimator(CommandLineTool):
    arguments = [Argument('test_file', short_opt='-i', type=str, required=True,
            help='the dataset in HDF5 format, required datasets: X_test, y_test'),
        Argument('model_file', type=str, required=True,
            help='file path for saving the model (in Python pickle format)'),
        Argument('model_type', type=str, default='sklearn', choices=('sklearn', 'keras')),
        Argument('metrics', type=list, default='accuracy'),
        Argument('metric_file', short_opt='-o', type=str, required=True),
        Argument('flatten', action='store_true', help='flatten the input dataset before applying the model'),]
    def __call__(self):
        import h5py
        import zipfile

        self.logger.info('load test dataset: ' + self.test_file)
        f = h5py.File(self.test_file, 'r')
        X_test = f['X_test'][:]
        y_test = f['y_test'][:]
        f.close()
        if self.flatten:
            self.logger.info('flatten the test data to dimension: (%d, %d)'%X_test.shape[:2])
            X_test = X_test.reshape((X_test.shape[0], -1))
        if self.model_type == 'keras':
            import_keras()
            self.logger.info('load keras model: ' + self.model_file)
            model = keras.models.load_model(self.model_file)
        elif self.model_type == 'sklearn':
            import cPickle
            self.logger.info('load sklearn model: ' + self.model_file)
            zipf = zipfile.ZipFile(self.model_file, 'r')
            f = zipf.open('model', 'r')
            model = cPickle.load(f)
            zipf.close()
        if self.model_type == 'sklearn':
            y_pred_labels = model.predict(X_test)
            model_name = model.__class__.__name__
            if model_name == 'SVC':
                y_pred = model.decision_function(X_test)
            elif model_name == 'RandomForestClassifier':
                y_pred = model.predict_proba(X_test)[:, 1]
            else:
                raise ValueError('unknown sklearn model ' + model_name)
        elif self.model_type == 'keras':
            y_pred = model.predict(X_test)
            y_pred_labels = (y_pred >= 0.5).astype('int32')

        self.logger.info('save metrics: ' + self.metric_file)
        prepare_output_file(self.metric_file)
        f = h5py.File(self.metric_file, 'w')
        f.create_dataset('y_true', data=y_test)
        f.create_dataset('y_pred', data=y_pred)
        f.create_dataset('y_pred_labels', data=y_pred_labels)
        g = f.create_group('metrics')
        for metric in self.metrics:
            scorer = get_scorer(metric)
            if metric == 'roc_auc':
                score = scorer(y_test, y_pred)
            else:
                score = scorer(y_test, y_pred_labels)
            self.logger.info('calculate metric {}: {}'.format(metric, score))
            g.create_dataset(metric, data=score)
        f.close()

class CreateCvIndex(CommandLineTool):
    arguments = [Argument('n_samples', short_opt='-n', type=int,
            help='number of samples'),
        Argument('data_file', short_opt='-i', type=str,
            help='determine the number of samples from input file (from y_train)'),
        Argument('n_folds', short_opt='-k', type=int, required=True,
            help='number of folds'),
        Argument('outfile', short_opt='-o', type=str, required=True,
            help='output file in HDF5 format with dataset names: /<fold>/train, /<fold>/test')]
    def __call__(self):
        import h5py
        from sklearn.model_selection import KFold
        import numpy as np
        if (self.n_samples is None) and (self.data_file is None):
            raise ValueError('either --n-samples/--data-file should be specified')
        if self.data_file:
            self.logger.info('determine number of samples from data file: {}' + self.data_file)
            fin = h5py.File(self.data_file, 'r')
            self.n_samples = fin['y_train'].shape[0]
            fin.close()
            self.logger.info('number of training samples: {}'.format(self.n_samples))
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
    arguments = [Argument('infile', short_opt='-i', default='-', type=str,
            help='parameter grid specification in JSON format'),
        Argument('grid_spec', short_opt='-s', type=str,
            help='parameter grid specification as command line arguments in JSON format'),
        Argument('sample', type=int,
            help='randomly sample up to a certain number of parameters'),
        Argument('outfile', short_opt='-o', type=str, default='-',
            help='output file')]
    def __call__(self):
        import json
        import itertools
        import random

        if self.grid_spec:
            grid_spec = json.loads(self.grid_spec)
        else:
            fin = open_file_or_stdin(self.infile)
            grid_spec = json.load(fin)
            fin.close()

        fout = open_file_or_stdout(self.outfile)
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
        fout.close()

class SelectBestModel(CommandLineTool):
    arguments = [Argument('cvdir', short_opt='-i', type=str, required=True,
            help='directory containing prediction metric files in HDF5 format with datasets: y_true, y_pred'),
        Argument('metric', type=str, default='accuracy'),
        Argument('prefix', short_opt='-o', type=str, required=True)]
    def __call__(self):
        import h5py
        import json
        import pandas as pd
        from sklearn.metrics import roc_auc_score, accuracy_score

        hyperparams = []
        with open(os.path.join(self.cvdir, 'hyperparam.txt'), 'r') as f:
            for line in f:
                hyperparams.append(line.strip())
        f = h5py.File(os.path.join(self.cvdir, 'cv_index.h5'), 'r')
        n_folds = len(f.keys())
        f.close()

        scores = []
        for param_index, hyperparam in enumerate(hyperparams):
            for cv_fold in range(n_folds):
                metric_file = '%s/%d/%d.valid_metrics'%(self.cvdir, param_index, cv_fold)
                f = h5py.File(metric_file, 'r')
                score = accuracy_score(f['y_pred_labels'][:], f['y_true'][:])
                scores.append((param_index, cv_fold, score, hyperparam))
        scores = pd.DataFrame.from_records(scores, columns=('param_index', 'cv_index', self.metric, 'hyperparam'))
        scores.to_csv(self.prefix + '.detail.txt', sep='\t', index=False, doublequote=False, quotechar="'")

        scores_by_hyperparam = scores.groupby(['param_index'], as_index=False)[self.metric].mean()
        scores_by_hyperparam['hyperparam'] = hyperparams
        scores_by_hyperparam.to_csv(self.prefix + '.mean_by_hyperparam.txt', sep='\t', index=False, doublequote=False, quotechar="'")
        best_param_index = scores_by_hyperparam['param_index'][scores_by_hyperparam[self.metric].idxmax()]

        with open(self.prefix + '.best_hyperparam.json', 'w') as f:
            f.write(hyperparams[best_param_index])
        with open(self.prefix + '.best_param_index.txt', 'w') as f:
            f.write(str(best_param_index))


if __name__ == '__main__':
    CommandLineTool.from_argv()()
