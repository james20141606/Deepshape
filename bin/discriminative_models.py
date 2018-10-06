import numpy as np
import h5py
from abc import ABC, abstractmethod

# import keras modules
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, Reshape
from keras.models import Model
from keras.layers.merge import Concatenate, Add
from keras.layers.convolutional import Conv1D, Conv2D, UpSampling1D, UpSampling2D
from keras.layers.pooling import MaxPooling1D, MaxPooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, AveragePooling1D
from keras.regularizers import l2, l1, l1_l2
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras import backend as K

class DiscriminativeModel(ABC):
    @abstractmethod
    def fit(self, X, y, sample_weight=None, **kwargs):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        pass

    def init_params(self, **kwargs):
        pass

class NeuralNetworkClassifier(DiscriminativeModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build()
        self.model.compile(loss='binary_crossentropy',
            metrics=['binary_accuracy'],
            optimizer='Adam')
    
    @abstractmethod
    def build(self):
        pass

    def fit(self, X, y, sample_weight=None, batch_size=64, epochs=10):
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, sample_weight=sample_weight)
    
    def predict(self, X):
        return np.greater_equal(self.predict_proba, 0.5).astype(np.int32)
    
    def predict_proba(self, X):
        return self.model.predict(X).flatten()
    
    def init_params(self):
        session = K.get_session()
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

    def save(self, filename):
        self.model.save(filename)
    
    def load(self, filename):
        self.model = keras.models.load_model(filename)

class DeepBind(NeuralNetworkClassifier):
    def build(self):
        model = Sequential()
        model.add(Conv1D(16, 24, padding='valid', input_shape=self.input_shape, kernel_regularizer=l2(0.0001), name='conv'))
        model.add(Activation('relu', name='relu'))
        model.add(GlobalMaxPooling1D(name='global_maxpool'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(32, kernel_regularizer=l2(0.0001), activation='sigmoid', name='dense1'))
        model.add(Dense(1, kernel_regularizer=l2(0.0001), name='dense2'))
        model.add(Activation('sigmoid', name='sigmoid'))
        return model

class IterativeEnrich(DiscriminativeModel):
    def __init__(self, estimator):
        self.estimator = None

    def fit(self, X, y=None, sample_weight=None, tol=1e-3, fit_kw=None, predict_kw=None, max_iter=300):
        if y is None:
            # assign labels to random values
            y = np.random.randint(2, size=X.shape[0], dtype=np.int32)
        else:
            y = y.flatten().astype(np.int32)
        if fit_kw is None:
            fit_kw = {}
        if predict_kw is None:
            predict_kw = {}
        y_prev = y
        i_iter = 0
        while i_iter < max_iter:
            self.estimator.fit(X, y_prev, **fit_kw)
            y_probas = self.estimator.predict_proba(X, **predict_kw)
            y_labels = (y_probas[:, 1] >= 0.5).astype(np.int32)
            error_rate = np.mean(~np.equal(y_labels, y_prev))
            i_iter += 1
            if error_rate < tol:
                break
            y_prev = y_labels
            #print('iter = {}, error rate = {}'.format(i_iter, error_rate))
        print('optimized in {} iterations'.format(i_iter))
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    
    def init_params(self, **kwargs):
        self.estimator.init_params(**kwargs)
