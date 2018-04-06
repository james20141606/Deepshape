#! /usr/bin/env python
import os, sys, argparse
from common import make_dir, get_scorer, get_scorer_type, onehot_encode
from formats import read_ct, read_fasta, read_rnafold
from cmdtool import CommandLineTool, Argument
from ioutils import prepare_output_file
import logging
logging.basicConfig(level=logging.DEBUG)

# import keras
execfile(os.path.join(os.path.dirname(__file__), 'import_keras.py'))

def plot_motif_detect(weblogo_file, model, alphabet, n_weights):
    # plot weights of the first convolution layer
    """
    import matplotlib
    matplotlib.use('Cairo')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, subplots = plt.subplots(nrows=n_weights)
    W = K.get_value(model.layers[0].W)
    for i in range(n_weights):
        subplots[i].matshow(W[:, :, :, i].squeeze().T, cmap=cm.Greys)
    fig.savefig(weights_file)
    """
    W = K.get_value(model.layers[0].W)
    for i in range(n_weights):
        pwm = W[:, :, :, i].squeeze()
        pwm = (pwm - pwm.min())/(pwm.max() - pwm.min())
        pwm_to_weblogo('%s.%d.png'%(weblogo_file, i + 1), pwm, alphabet)
    with open(weblogo_file + '.txt', 'w') as f:
        for i in range(n_weights):
            f.write('W[%d]\n'%i)
            np.savetxt(f, W[:, :, :, i].squeeze())

def visualize_motif_detect(model, image_file,
        layer_name=None, n_steps=20, max_n_filters=64):
    from scipy.misc import imsave

    def normalize_input(x):
        if K.image_dim_ordering() == 'th':
            return x / np.expand_dims(x.sum(axis=0), 1)
        else:
            return x / np.expane_dims(x.sum(axis=1), 1)

    def tensor_to_image(x):
        x = (x - x.mean()) / (x.std() + 1e-5)
        x *= 0.1
        x += 0.5
        if K.image_dim_ordering() == 'th':
            x /= np.expand_dims(x.sum(axis=0), 1)
        else:
            x /= np.expand_dims(x.sum(axis=1), 1)
        x = np.clip(x, 0, 1)

        x *= 255
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    found = False
    for layer in model.layers:
        if layer.name == layer_name:
            found = True
            break
    if not found:
        raise ValueError('layer name %s is not found'%layer_name)

    if K.image_dim_ordering() == 'th':
        n_filters = K.int_shape(layer.output)[1]
        seq_length = K.int_shape(model.input)[2]
    else:
        n_filters = K.int_shape(layer.output)[2]
        seq_length = K.int_shape(model.input)[1]
    n_filters = min(n_filters, max_n_filters)

    kept_filters = []
    for filter_index in range(n_filters):
        print 'Learning visualization of filter %s[%d]'%(layer_name, filter_index)
        if K.image_dim_ordering() == 'th':
            loss = K.mean(layer.output[:, filter_index, :])
        else:
            loss = K.mean(layer.output[:, :, filter_index])
        grads = K.gradients(loss, model.input)[0]
        grads = grads / (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        iterate = K.function([model.input, K.learning_phase()], [loss, grads])
        # generate a random input sample with uniform initlaization
        input_data = np.random.random((1, K.int_shape(model.input)[1], K.int_shape(model.input)[2]))
        # gradient descent
        for i in range(n_steps):
            loss_value, grads_value = iterate([input_data, True])
            input_data += grads_value
            # skip filters that get stuck at 0
            if loss_value <= 0.0:
                break
        # decode the data
        if loss_value > 0:
            input_data = tensor_to_image(input_data[0])
            kept_filters.append((input_data, loss_value))
    # only keep filters with high loss
    kept_filters.sort(key=lambda x: x[1], reverse=True)

    n = len(kept_filters)
    margin = 1
    scale_factor = 10
    img_height = seq_length*scale_factor
    img_width = (n*4 + (n - 1)*margin)*scale_factor
    img = np.full((img_width, img_height), 255, dtype='uint8')

    for filter_index in range(n):
        input_data, loss_value = kept_filters[filter_index]
        for i in range(4):
            for j in range(seq_length):
                img[((4 + margin)*filter_index + i)*scale_factor : ((4 + margin)*filter_index + i + 1)*scale_factor,
                    j*scale_factor : (j+1)*scale_factor] = input_data[j, i]
    imsave(image_file, img)

class TrainDeepfold1D(CommandLineTool):
    description = 'Train a Deepfold model'
    arguments = [Argument('infile', short_opt='-i', type=str, required=True, help='input training data'),
        Argument('model_script', type=str, required=True, help='a Python script that create a Keras model object'),
        Argument('model_file', type=str, required=True, help='file path for saving the model'),
        Argument('batch_size', type=int, default=50),
        Argument('epochs', type=int, default=20),
        Argument('learning_rate', type=float, default=0.0005),
        Argument('regression', action='store_true'),
        Argument('xname', type=str, default='X_train', help='training data matrix name'),
        Argument('yname', type=str, default='y_train', help='training target name'),
        Argument('tensorboard_log_dir', type=str, help='output directory for TensorBoard'),
        Argument('keras_verbose', type=int, default=1, help='verbosity in the keras model.fit() function'),
        Argument('keras_log', type=str, help='CSV log file'),
        Argument('valid_file', type=str, help='validation data'),
        Argument('valid_xname', type=str, default='X_valid'),
        Argument('valid_yname', type=str, default='y_valid')]

    def __call__(self):
        import h5py

        self.logger.info('load training data: ' + self.infile)
        fin = h5py.File(self.infile, 'r')
        X_train = fin[self.xname][:]
        y_train = fin[self.yname][:]
        fin.close()

        valid_data = None
        if self.valid_file:
            self.logger.info('load validation data: ' + self.valid_file)
            fin = h5py.File(self.valid_file, 'r')
            X_valid = fin[self.valid_xname][:]
            y_valid = fin[self.valid_yname][:]
            fin.close()
            valid_data = (X_valid, y_valid)

        window_size = X_train.shape[1]
        from keras.optimizers import RMSprop
        optimizer = RMSprop(lr=self.learning_rate)
        # load model
        # variables optimizer, loss may be overloaded
        regression = self.regression
        if self.regression:
            loss = 'mean_squared_error'
            metrics = ['mean_squared_error']
        else:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']

        with open(self.model_script, 'r') as f:
            exec compile(f.read(), self.model_script, 'exec')

        model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=metrics)
        model.summary()

        callbacks = []
        if self.tensorboard_log_dir:
            from keras.callbacks import TensorBoard
            callbacks = [TensorBoard(log_dir=self.tensorboard_log_dir)]
        else:
            callbacks = []
        if self.keras_log is not None:
            self.logger.info('open CSV log file: {}'.format(self.keras_log))
            make_dir(os.path.dirname(self.keras_log))
            callbacks.append(keras.callbacks.CSVLogger(self.keras_log))

        self.logger.info('train model')
        model.fit(X_train, y_train,
            batch_size=self.batch_size, epochs=self.epochs,
            callbacks=callbacks, verbose=self.keras_verbose,
            validation_data=valid_data)
        self.logger.info('save model: {}'.format(self.model_file))
        prepare_output_file(self.model_file)
        model.save(self.model_file)

class EvaluateDeepfold1D(CommandLineTool):
    description = 'Train a Deepfold model'
    arguments = [Argument('infile', short_opt='-i', type=str, required=True, help='input training data'),
        Argument('model_file', type=str, required=True, help='file path for saving the model'),
        Argument('model_format', type=str, default='keras', choices=('keras', 'sklearn')),
        Argument('outfile', short_opt='-o', type=str, help='output file to write the prediction and metrics to'),
        Argument('batch_size', type=int, default=200),
        Argument('swap_labels', action='store_true', help='swap 0/1 predictions'),
        Argument('metrics', type=list, default='accuracy', help='a metric name defined in sklearn.metrics.get_scorer'),
        Argument('cutoff', type=float, default=0.5, help='cutoff for converting predictions to binary labels'),
        Argument('xname', type=str, default='X_test', help='dataset name of the test data matrix'),
        Argument('yname', type=str, default='y_test', help='dataset name of the test labels')]
    def __call__(self):
        import h5py
        self.logger.info('load model: {}'.format(self.model_file))
        if self.model_format == 'keras':
            model = keras.models.load_model(self.model_file)
        elif self.model_format == 'sklearn':
            import cPickle
            with open(self.model_file, 'r') as f:
                model = cPickle.load(f)

        self.logger.info('load data: {}'.format(self.infile))
        fin = h5py.File(self.infile, 'r')
        X_test = fin[self.xname][:]
        y_test = fin[self.yname][:]
        fin.close()

        self.logger.info('run the model')
        if self.model_format == 'keras':
            y_pred = model.predict(X_test, batch_size=self.batch_size)
        elif self.model_format == 'sklearn':
            y_pred = model.predict(X_test)

        y_pred = np.squeeze(y_pred)
        if self.swap_labels:
            self.logger.info('swap labels')
            y_pred = 1 - y_pred
        y_pred_labels = (y_pred >= self.cutoff).astype('int32')

        # ingore NaNs in y_test
        y_test = y_test.flatten()
        y_pred = y_pred.flatten()
        y_pred_labels = y_pred_labels.flatten()
        not_nan_mask = np.logical_not(np.isnan(y_test))
        y_test = y_test[not_nan_mask]
        y_pred = y_pred[not_nan_mask]
        y_pred_labels = y_pred_labels[not_nan_mask]

        scores = {}
        for metric in self.metrics:
            # y_pred is an array of continous scores
            scorer = get_scorer(metric)
            if metric == 'roc_auc':
                scores[metric] = scorer(y_test, y_pred)
            else:
                scores[metric] = scorer(y_test, y_pred_labels)
            self.logger.info('metric {} = {}'.format(metric, scores[metric]))
        if self.outfile is not None:
            self.logger.info('save file: {}'.format(self.outfile))
            make_dir(os.path.dirname(self.outfile))
            fout = h5py.File(self.outfile, 'w')
            fout.create_dataset('y_true', data=y_test)
            fout.create_dataset('y_pred', data=y_pred)
            fout.create_dataset('y_pred_labels', data=y_pred_labels)
            grp = fout.create_group('metrics')
            for metric in self.metrics:
                grp.create_dataset(metric, data=scores[metric])
            fout.close()

class MutateMap(CommandLineTool):
    description = ''
    arguments = [Argument('model_file', type=str, required=True),
        Argument('n_sequences', type=int, default=100)]
    def __call__(self):
        import numpy as np

        self.logger.info('load model: {}'.format(self.model_file))
        model = keras.models.load_model(self.model_file)
        window_size = K.int_shape(model.input)[1]
        n = K.int_shape(model.input)[2]

        X = np.empty((n_sequences*(n - 1)*window_size, window_size), dtype='float32')
        X_wt = np.random.randint(n, size=(n_sequences, window_size))
        X_mut = np.repeat(X_wt, self.n_sequences*(window_size*(n - 1)))
        for i in range(self.n_sequences):
            for j in range(n):
                X_mut[j::n] += (j + 1)
            X_mut = np.mod(X_mut, n)
        X_wt = onehot_encode(X_wt, range(n))
        X_mut = onehot_encode(X_mut, range(n))
        y_wt = model.predict(X_wt)
        y_mut = model.predict(X_mut)

class PredictDeepfold1D(CommandLineTool):
    description = 'Predict 1D profile from sequence files'
    arguments = [Argument('infile', short_opt='-i', type=str, required=True,
            help='sequence file in FASTA format'),
        Argument('format', type=str, default='fasta',
            choices=('ct', 'ct_dir', 'fasta', 'rnafold', 'genomic_data'),
            help='input file format (fasta or ct)'),
        Argument('model_file', type=str, required=True,
            help='file path for saving the model'),
        Argument('offset', type=int, required=False,
            help='offset of the base in the window to predict'),
        Argument('pred_file', short_opt='-o', type=str,
            help='output file to write the prediction to'),
        Argument('cutoff', type=float, default=0.5,
            help='cutoff for converting predictions to binary labels'),
        Argument('batch_size', type=int, default=200),
        Argument('dense', action='store_true'),
        Argument('metrics', type=list, default='accuracy,roc_auc,ppv,sensitivity',
            help='a metric name defined in sklearn.metrics.get_scorer'),
        Argument('fillna', type=float,
            help='fill missing values with a constant'),
        Argument('alphabet', default='ATCG'),
        Argument('swap_labels', action='store_true',
            help='swap 0/1 predictions'),
        Argument('split', action='store_true',
            help='output separate files in the output directory'),
        Argument('restraint_file', type=str,
            help='restraint file for RME'),
        Argument('metric_file', type=str,
            help='prediction scores and metrics'),
        Argument('metric_by_sequence_file', type=str,
            help='a text table with metrics calculated for each sequence'),
        Argument('dense_pred_file', type=str,
            help='output dense predictions')
    ]

    def sequence_to_windows(self, seq, window_size, offset):
        """
        Convert a sequence to fixed length short sequences
        Returns: a list of sequences with length 'window_size'
        """
        pad_seq = ''.join(['N']*window_size)
        windows = []
        for i in range(len(seq)):
            seq_start = max(0, i - offset)
            seq_end = min(i - offset + window_size, len(seq))
            win_start = offset - (i - seq_start)
            win_end = offset + (seq_end - i)
            window = bytearray(pad_seq)
            seq_win = seq[seq_start:seq_end]
            window[win_start:win_end] = seq_win
            window = str(window)
            windows.append(window)
        return windows

    def predict_dense(self, y_pred, offset):
        import numpy as np
        seq_length, window_size = y_pred.shape
        y_pred_seq = np.zeros(y_pred.shape[0], dtype='float32')
        scale = np.zeros(y_pred.shape[0], dtype='float32')
        for i in range(y_pred.shape[0]):
            seq_start = max(0, i - offset)
            seq_end = min(i - offset + window_size, seq_length)
            win_start = offset - (i - seq_start)
            win_end = offset + (seq_end - i)
            y_pred_seq[seq_start:seq_end] += y_pred[i][win_start:win_end]
            scale[seq_start:seq_end] += 1.0
        y_pred_seq /= scale
        return y_pred_seq

    def __call__(self):
        import numpy as np
        import pandas as pd
        import h5py
        from formats import read_rnafold, structure_to_pairs

        self.logger.info('load model: {}'.format(self.model_file))
        model = keras.models.load_model(self.model_file)
        window_size = K.int_shape(model.input)[1]
        self.logger.info('load input data (in %s format): %s'%(self.format, self.infile))
        have_structure = False
        if self.format == 'fasta':
            # list of tuples: (name, seq)
            input_data = list(read_fasta(self.infile))
        elif self.format == 'ct_dir':
            # read all .ct files from the directory
            # list of tuples: (name, seq, pairs)
            input_data = []
            for filename in os.listdir(self.infile):
                title, seq, pairs = read_ct(os.path.join(self.infile, filename))
                title = os.path.splitext(filename)[0]
                input_data.append((title, seq, pairs))
            have_structure = True
        elif self.format == 'ct':
            title, seq, pairs = read_ct(self.infile)
            title = os.path.splitext(os.path.basename(self.infile))[0]
            input_data = [(title, seq, pairs)]
            have_structure = True
        elif self.format == 'rnafold':
            input_data = []
            for name, seq, structure, energy in read_rnafold(self.infile, parse_energy=False):
                pairs = structure_to_pairs(structure)
                input_data.append((name, seq, pairs))
            have_structure = True
        elif self.format == 'genomic_data':
            from genomic_data import GenomicData
            input_data = []
            data = GenomicData(self.infile)
            for name in data.names:
                input_data.append((name,
                    data.feature('sequence', name).tostring(),
                    data.feature('reactivity', name)))
            del data
            have_structure = True

        # combine all structures (base-pairs) into one array in the ct file
        if have_structure:
            structure = []
            for i in range(len(input_data)):
                structure.append(np.asarray(input_data[i][2], dtype='int32'))
            structure = np.concatenate(structure)
        else:
            structure = None

        X = []
        names = []
        # offset default to the center of the window
        if self.offset is None:
            self.offset = (window_size + 1)/2
        offset = self.offset

        # convert sequences to windows
        windows = []
        length = []
        sequence = []
        for item in input_data:
            name = item[0]
            seq = item[1]
            windows += self.sequence_to_windows(seq, window_size, offset)
            names.append(name)
            length.append(len(seq))
            sequence.append(seq)
        # combine all sequences into one dataset
        sequence = np.frombuffer(''.join(sequence), dtype='S1')
        length = np.asarray(length, dtype='int64')

        n_samples = len(windows)
        windows = np.frombuffer(''.join(windows), dtype='S1').reshape((n_samples, window_size))
        X = onehot_encode(windows, self.alphabet)
        # set one-hot coding of padded sequence to [0.25, 0.25, 0.25, 0.25]
        X[X.sum(axis=2) == 0] = 1.0/len(self.alphabet)

        self.logger.info('run the model')
        y_pred = model.predict(X, batch_size=self.batch_size)
        y_pred = np.squeeze(y_pred)
        if self.swap_labels:
            self.logger.info('swap labels')
            y_pred = 1 - y_pred

        # start/end position of each transcript in the y_pred
        end = np.cumsum(length)
        start = end - length
        if len(y_pred.shape) > 1:
            # average the predictions
            self.logger.info('average windows for dense prediction')
            y_pred_dense = []
            for i in range(len(input_data)):
                y_pred_dense.append(self.predict_dense(y_pred[start[i]:end[i]], offset))

            if self.dense_pred_file:
                self.logger.info('save dense predictions: ' + self.dense_pred_file)
                f = h5py.File(self.dense_pred_file, 'w')
                for i in range(len(names)):
                    g = f.create_group(names[i])
                    g.create_dataset('predicted_values_dense', data=y_pred[start[i]:end[i]])
                    g.create_dataset('predicted_values_average', data=y_pred_dense[i])
                    # 0-based start/end position of each transcript in the array (y_pred, sequence, structure)
                    g.create_dataset('sequence', data=sequence[start[i]:end[i]])
                    if structure is not None:
                        g.create_dataset('structure', data=structure[start[i]:end[i]])
                f.close()

            y_pred = np.concatenate(y_pred_dense)
            y_pred_labels = np.round(y_pred).astype('int32')
        else:
            y_pred_labels = np.round(y_pred).astype('int32')

        if self.restraint_file:
            header = ['name', 'position', 'pred', 'base']
            table = pd.DataFrame()
            table['name'] = np.repeat(np.asarray(names, dtype='S'), length)
            # start position of each transcript relative to the y_pred
            start = np.repeat(cum_length - length, length)
            # position (1-based) relative to the transcript
            position = np.arange(1, length.sum() + 1) - start
            table['position'] = position
            table['pred'] = y_pred_labels
            table['base'] = sequence
            table['true'] = structure
            self.logger.info('write restraint file: ' + self.restraint_file)
            prepare_output_file(self.restraint_file)
            table.to_csv(self.restraint_file, sep='\t', index=False)
        if self.metric_file:
            self.logger.info('save metric file: ' + self.metric_file)
            prepare_output_file(self.metric_file)
            f = h5py.File(self.metric_file, 'w')
            from sklearn.metrics import accuracy_score
            f.create_dataset('y_pred', data=y_pred)
            f.create_dataset('y_pred_labels', data=y_pred_labels)
            if have_structure:
                #print structure
                y_true = (structure > 0).astype('int32')
                f.create_dataset('y_true', data=y_true)
                g = f.create_group('metrics')
                for metric in self.metrics:
                    scorer = get_scorer(metric)
                    if get_scorer_type(metric) == 'continous':
                        score = scorer(y_true, y_pred)
                    else:
                        score = scorer(y_true, y_pred_labels)
                    self.logger.info('%s: %f'%(metric, score))
                    g.create_dataset(metric, data=score)
            f.close()
        if self.metric_by_sequence_file:
            self.logger.info('calculate metrics by sequence')
            records = []
            for i in range(len(names)):
                y_true_ = (structure[start[i]:end[i]] > 0).astype('int32')
                y_pred_ = y_pred[start[i]:end[i]]
                y_pred_labels_ = y_pred_labels[start[i]:end[i]]
                scores = []
                for metric in self.metrics:
                    scorer = get_scorer(metric)
                    if get_scorer_type(metric) == 'continuous':
                        try:
                            score = scorer(y_true_, y_pred_)
                        except ValueError:
                            score = np.nan
                    else:
                        score = scorer(y_true_, y_pred_labels_)
                    scores.append(score)
                records.append([names[i], length[i]] + scores)
            records = pd.DataFrame.from_records(records, columns=['name', 'length'] + self.metrics)
            self.logger.info('save metric by sequence file: ' + self.metric_by_sequence_file)
            prepare_output_file(self.metric_by_sequence_file)
            records.to_csv(self.metric_by_sequence_file, sep='\t', index=False, na_rep='nan')
        if self.pred_file:
            self.logger.info('save predictions to file: ' + self.pred_file)
            prepare_output_file(self.pred_file)
            f = h5py.File(self.pred_file, 'w')
            for i in range(len(names)):
                y_true_ = (structure[start[i]:end[i]] > 0).astype('int32')
                g = f.create_group(names[i])
                g.create_dataset('sequence', data=sequence[start[i]:end[i]])
                g.create_dataset('predicted_values', data=y_pred[start[i]:end[i]])
                g.create_dataset('predicted_labels', data=y_pred[start[i]:end[i]])
                g.create_dataset('true_labels', data=y_true_)
            f.close()

class EvaluateRnafold(CommandLineTool):
    description = 'Evaluate 1D prediction of RNAfold'
    arguments = [Argument('infile', short_opt='-i', type=str, required=True, help='RNAfold output file'),
        Argument('known_file', type=str, required=True, help='known structure in GenomicData format'),
        Argument('outfile', short_opt='-o', type=str, help='output file to write the prediction and metrics to'),
        Argument('feature', type=str, required=True, help='feature name in known-file'),
        Argument('metrics', type=list, default='accuracy,sensitivity,ppv')]
    def __call__(self):
        from genomic_data import GenomicData
        import pandas as pd
        import numpy as np

        known = GenomicData(self.known_file, [self.feature])
        y_pred = []
        y_true = []
        names = []
        length = []
        for name, seq, structure, energy in read_rnafold(self.infile):
            names.append(name)
            structure = np.frombuffer(structure, dtype='S1')
            length.append(len(structure))
            y_pred.append((structure != '.').astype('int32'))
            y_true_seq = known.feature(self.feature, name)
            if known.feature(self.feature, name) is None:
                found = np.nonzero(map(lambda x: x.startswith(name), known.names))[0]
                if len(found) == 0:
                    raise ValueError('sequence {} could not be found'.format(name))
                elif len(found) == 1:
                    self.logger.warn('partial sequence name match {} => {}'.format(known.names[found[0]], name))
                    y_true_seq = known.feature(self.feature, known.names[found[0]])
                else:
                    raise ValueError('multiple partial matches found for {}'.format(name))
            y_true.append(y_true_seq)
        """
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

        scores = {}
        for metric in self.metrics:
            # y_pred is an array of continous scores
            scorer = get_scorer(metric)
            scores[metric] = scorer(y_true, y_pred)
            self.logger.info('metric {} = {}'.format(metric, scores[metric]))
        if self.outfile is not None:
            self.logger.info('save file: {}'.format(self.outfile))
            prepare_output_file(self.outfile)
            fout = h5py.File(self.outfile, 'w')
            fout.create_dataset('y_true', data=y_true)
            fout.create_dataset('y_pred', data=y_pred)
            fout.create_dataset('y_pred_labels', data=y_pred)
            grp = fout.create_group('metrics')
            for metric in self.metrics:
                scorer = get_scorer(metric)
                if get_scorer_type(metric) == 'continuous':
                    try:
                        score = scorer(y_true, y_pred)
                    except ValueError:
                        score = np.nan
                else:
                    score = scorer(y_true, y_pred_labels)
                
                grp.create_dataset(metric, data=scores[metric])
            fout.close()"""
        if True:
            self.logger.info('calculate metrics by sequence')
            records = []
            for i in range(len(names)):
                y_true_ = y_true[i]
                y_pred_ = y_pred[i]
                y_pred_labels_ = y_pred_
                scores = []
                for metric in self.metrics:
                    scorer = get_scorer(metric)
                    if get_scorer_type(metric) == 'continuous':
                        try:
                            score = scorer(y_true_, y_pred_)
                        except ValueError:
                            score = np.nan
                    else:
                        score = scorer(y_true_, y_pred_labels_)
                    scores.append(score)
                records.append([names[i], length[i]] + scores)
            records = pd.DataFrame.from_records(records, columns=['name', 'length'] + self.metrics)
            self.logger.info('save metric by sequence file: ' + self.outfile)
            prepare_output_file(self.outfile)
            records.to_csv(self.outfile, sep='\t', index=False, na_rep='nan')

class DrawModelOutputImages(CommandLineTool):
    arguments = [Argument('model_file', short_opt='-i', type=str, required=True),
                 Argument('output_dir', short_opt='-o', type=str, required=True),
                 Argument('n_images', type=int, default=3)]
    def write_model_output_images(self, model, X, output_dir, n_images=3):
        import numpy as np
        import h5py
        from scipy.misc import imsave
        import keras.backend as K
        from keras.utils.vis_utils import model_to_dot
        import subprocess

        def zoom_image(x, scale=(1, 1), separator=(0, 0)):
            x = np.repeat(x, scale[0] + separator[0], axis=0)
            x = np.repeat(x, scale[1] + separator[1], axis=1)
            if separator[0] > 0:
                x[scale[0]::(scale[0] + separator[0]), :] = 0
            if separator[1] > 0:
                x[:, scale[1]::(scale[1] + separator[1])] = 0
            if separator[0] > 0:
                x = x[:-separator[0], :]
            if separator[1] > 0:
                x = x[:, :-separator[1]]
            return x

        input_seq = model.input
        layer_outputs = []
        layer_names = []
        for layer in model.layers:
            if hasattr(layer, 'output'):
                layer_outputs.append(layer.output)
                layer_names.append(layer.name)
        get_images = K.function([K.learning_phase(), input_seq], layer_outputs)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # convert layer outputs to images
        for i in range(min(n_images, X.shape[0])):
            input_seq_val = np.expand_dims(X[i], axis=0)
            images = get_images([False, input_seq_val])

            hdf5_file = os.path.join(output_dir, '%d.data' % i)
            h5f = h5py.File(hdf5_file, 'w')
            scaled_images = {}
            for name, image in zip(layer_names, images):
                image = np.asarray(image)
                h5f.create_dataset(name, data=image)
                if name == 'input':
                    image = np.squeeze(image, axis=0).T
                    image = zoom_image(image, scale=(10, 3))
                elif name.startswith('index'):
                    image = np.squeeze(image, axis=0)
                    image = zoom_image(image, scale=(50, 3))
                elif (len(image.shape) == 3):
                    if image.shape[1] == 1:
                        image = image.reshape((1, -1))
                        image = zoom_image(image, (50, 3))
                    else:
                        image = np.squeeze(image, axis=0).T
                        image = zoom_image(image, scale=(3, 3))
                elif len(image.shape) == 2:
                    image = image.reshape((1, -1))
                    image = zoom_image(image, (50, 3))
                elif len(image.shape) == 0:
                    image = image.reshape((1, 1))
                    image = zoom_image(image, (50, 50))
                else:
                    continue
                scaled_images[name] = image
            images = scaled_images

            g = model_to_dot(model)
            image_dir = os.path.join(output_dir, '%d.images' % i)
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            dpi = 96.0
            for node in g.get_nodes():
                label = node.get_label()
                if not label:
                    continue
                label = label.split(':')[0]
                image = images.get(label)
                if image is None:
                    continue
                # image = (image - image.min())/(image.max() - image.min())
                image_file = os.path.join(image_dir, '%s.png' % label)
                imsave(image_file, image)
                # node.set_image(image_file)
                node.set_image('%d.images/%s.png' % (i, label))
                node.set_width(image.shape[1] / dpi + 0.5)
                node.set_height(image.shape[0] / dpi + 1.5)
                node.set_labelloc('b')
                node.set_shape('box')
            g.set_rankdir('LR')
            g.write(os.path.join(output_dir, '%d.gv' % i), format='raw')
            p = subprocess.Popen(['dot', '-Tsvg', '-o',
                                  '%d.svg' % i,
                                  '%d.gv' % i],
                                 cwd=output_dir)
            p.communicate()

    def __call__(self):
        import keras
        import keras.backend as K
        import numpy as np
        self.logger.info('load model: {}'.format(self.model_file))
        model = keras.models.load_model(self.model_file)
        window_size = K.int_shape(model.input)[1]

        alphabet = np.frombuffer('ATCG', dtype='S1')
        sequences = np.random.randint(4, size=(self.n_images, window_size))
        sequences = np.take(alphabet, sequences)
        X = onehot_encode(sequences, 'ATCG')
        self.logger.info('write output images: ' + self.output_dir)
        self.write_model_output_images(model, X, self.output_dir, self.n_images)

if __name__ == "__main__":
    tool = CommandLineTool.from_argv()
    import_keras()
    patch_keras()
    tool()
