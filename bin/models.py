from __future__ import print_function
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import keras
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
from functools import partial


class ModelFunction(object):
    def __call__(self, window_size, n_channels=4, regression=False, dense=False):
        raise NotImplementedError('class ModelFunction must be subclassed')

class basic(ModelFunction):
    def __call__(self, window_size, n_channels=4, regression=False, dense=False):
        model = Sequential()
        model.add(Flatten(input_shape=(window_size, n_channels)))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        output_size = window_size if dense else 1
        model.add(Dense(output_size))
        if not regression:
            model.add(Activation('sigmoid'))
        return model

class conv1(ModelFunction):
    def __call__(self, window_size, n_channels=4, regression=False, dense=False):
        model = Sequential()
        model.add(Conv1D(64, 5, padding='valid', input_shape=(window_size, n_channels), kernel_regularizer=l2(0.0001)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(64, 4, padding='valid', kernel_regularizer=l2(0.0001)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(64, 4, padding='valid', kernel_regularizer=l2(0.0001)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dense(64, kernel_regularizer=l2(0.0001)))
        model.add(Activation('relu'))
        output_size = window_size if dense else 1
        model.add(Dense(output_size, kernel_regularizer=l2(0.0001)))
        if not regression:
            model.add(Activation('sigmoid'))
        return model

class conv1_motif(ModelFunction):
    def __call__(self, window_size, n_channels=4, regression=False, dense=False):
        model = Sequential()
        model.add(Conv1D(64, 7, padding='valid', input_shape=(window_size, n_channels), kernel_regularizer=l2(0.0001)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(64, 3, padding='valid', kernel_regularizer=l2(0.0001)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(64, 3, padding='valid', kernel_regularizer=l2(0.0001)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dense(64, kernel_regularizer=l2(0.0001)))
        model.add(Activation('relu'))
        output_size = window_size if dense else 1
        model.add(Dense(output_size, kernel_regularizer=l2(0.0001)))
        if not regression:
            model.add(Activation('sigmoid'))
        return model

class mlp1(ModelFunction):
    def __call__(self, window_size, n_channels=4, regression=False, dense=False):
        model = Sequential()
        model.add(Flatten(input_shape=(window_size, n_channels)))
        model.add(Dense(64, kernel_regularizer=l2(0.0001)))
        model.add(Activation('relu'))
        model.add(Dense(64, kernel_regularizer=l2(0.0001)))
        model.add(Activation('relu'))
        output_size = window_size if dense else 1
        model.add(Dense(output_size, kernel_regularizer=l2(0.0001)))
        if not regression:
            model.add(Activation('sigmoid'))
        return model

class resnet1(ModelFunction):
    def __call__(self, window_size, n_channels=4, regression=False, dense=False):
        input = Input(shape=(window_size, n_channels))
        x = Conv1D(64, n_channels, padding='same', activation='relu')(input)
        for i in range(5):
            output = Conv1D(64, 3, padding='same', activation='relu')(x)
            output = Conv1D(64, 3, padding='same')(output)
            output = Add()([x, output])
            output = Activation('relu')(output)
            x = output
        output = AveragePooling1D(2)(output)
        output = Flatten()(output)
        output_size = window_size if dense else 1
        output = Dense(output_size)(output)
        if not regression:
            output = Activation('sigmoid')(output)
        model = Model(inputs=[input], outputs=[output])
        return model

class blstm3(ModelFunction):
    def __call__(self, window_size, n_channels=4, regression=False, dense=False):
        model = Sequential()
        model.add(Bidirectional(LSTM(150, return_sequences=True), input_shape=(window_size, n_channels), merge_mode='ave'))
        model.add(Bidirectional(LSTM(1, return_sequences=True), merge_mode='ave'))
        model.add(Flatten())
        output_size = window_size if dense else 1
        model.add(Dense(output_size, kernel_regularizer=l2(0.0001)))
        if regression:
            model.add(Activation('sigmoid'))
        return model

class MotifClassifier(object):
    def __init__(self, window_size=512, n_classes=2, n_channels=4, n_conv_layers=1):
        input_layer = Input(shape=(window_size, n_channels), name='input')
        output = input_layer
        if n_conv_layers >= 1:
            output = Conv1D(64, 8, padding='same', name='conv1')(output)
            output = Activation('relu', name='relu1')(output)
        if n_conv_layers >= 2:
            output = MaxPooling1D(4, name='maxpool1')(output)
            output = Conv1D(128, 3, padding='same', name='conv2')(output)
            output = Activation('relu', name='relu2')(output)
        if n_conv_layers >= 3:
            output = MaxPooling1D(4, name='maxpool2')(output)
            output = Conv1D(128, 3, padding='same', name='conv3')(output)
            output = Activation('relu', name='relu3')(output)
        if n_conv_layers >= 4:
            output = MaxPooling1D(4, name='maxpool3')(output)
            output = Conv1D(128, 3, padding='same', name='conv4')(output)
            output = Activation('relu', name='relu4')(output)
        if n_conv_layers > 0:
            output = GlobalMaxPooling1D(name='global_maxpool')(output)
        else:
            output = Flatten(name='flatten')(output)
        if n_classes == 2:
            output = Dense(1, name='dense1')(output)
            output = Activation('sigmoid', name='output')(output)
            model = Model(inputs=[input_layer], outputs=[output])
            model.compile(loss='binary_crossentropy',
                      metrics=['binary_accuracy'],
                      optimizer='Adam')
        elif n_classes > 2:
            output = Dense(n_classes, name='dense1')(output)
            output = Activation('softmax', name='output')(output)
            model = Model(inputs=[input_layer], outputs=[output])
            model.compile(loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'],
                      optimizer='Adam')
        self.model = model

def random_multinomial(p):
    '''Draw a sample from multinomial distributions
    Args:
        p: multinomial probabilities. The last dimension is the label.
    Returns:
        Samples drawn from each distribution in one-hot encoding.
    '''
    p_cum = np.cumsum(p, axis=-1)
    n_classes = p.shape[-1]
    q = np.random.uniform(size=p.shape)
    X_labels = np.argmax(q < p_cum, axis=-1)
    X = (np.expand_dims(X_labels, axis=-1) == np.arange(n_classes).reshape([1]*(len(p.shape) - 1) + [-1])).astype(np.int32)
    return X

def batch_generator(X, batch_size=25):
    '''Create a iterator that generates batches
    '''
    n_samples = X.shape[0]
    for i in range(n_samples//batch_size):
        yield X[(i*batch_size):((i + 1)*batch_size)]
    n_remains = n_samples%batch_size
    if n_remains > 0:
        yield X[-n_remains:]

class MotifVariationalAutoencoder(object):
    def build_encoder(self, input_layer, name='conv', n_layers=1):
        if name == 'conv':
            encoder = Conv1D(64, 8, padding='same', activation='relu', name='encoder_conv1')(input_layer)
            if n_layers >= 2:
                encoder = MaxPooling1D(4, name='encoder_maxpool1')(encoder)
                encoder = Conv1D(128, 3, padding='same', activation='relu', name='encoder_conv2')(encoder)
            if n_layers >= 3:
                encoder = MaxPooling1D(4, name='encoder_maxpool2')(encoder)
                encoder = Conv1D(256, 3, padding='same', activation='relu', name='encoder_conv3')(encoder)
            encoder = GlobalMaxPooling1D(name='encoder_global_maxpool')(encoder)
        elif name == 'mlp':
            encoder = Flatten(name='encoder_flatten')(input_layer)
            encoder = Dense(128, name='encoder_dense1', activation='relu')(encoder)
            if n_layers == 2:
                encoder = Dense(128, name='encoder_dense2', activation='relu')(encoder)
        elif name == 'lstm':
            encoder = LSTM(64, name='encoder_lstm1', return_sequences=True)(encoder)
            if n_layers == 2:
                encoder = LSTM(64, name='encoder_lstm2', return_sequences=True)(encoder)
            encoder = Flatten(name='encoder_flatten')(encoder)
        return encoder

    def build_decoder(self, latent, name='conv', n_layers=1):
        decoder = latent
        if name == 'conv':
            if n_layers == 3:
                decoder = Dense(self.window_size//64*256, activation='relu', name='decoder_dense1')(decoder)
                decoder = Reshape((self.window_size//64, 256), name='decoder_reshape1')(decoder)
                decoder = UpSampling1D(4, name='decoder_upsample1')(decoder)
                decoder = Conv1D(128, 3, padding='same', activation='relu', name='decoder_conv1')(decoder)
                decoder = UpSampling1D(4, name='decocer_upsample2')(decoder)
                decoder = Conv1D(64, 3, padding='same', activation='relu', name='decoder_conv2')(decoder)
            if n_layers == 2:
                decoder = Dense(self.window_size//16*128, activation='relu', name='decoder_dense1')(decoder)
                decoder = Reshape((self.window_size//16, 128), name='decoder_reshape1')(decoder)
                decoder = UpSampling1D(4, name='decocer_upsample2')(decoder)
                decoder = Conv1D(64, 3, padding='same', activation='relu', name='decoder_conv2')(decoder)
            if n_layers == 1:
                decoder = Dense(self.window_size//4*64, activation='relu', name='decoder_dense1')(decoder)
                decoder = Reshape((self.window_size//4, 64), name='decoder_reshape1')(decoder)
            decoder = UpSampling1D(4, name='decoder_upsample3')(decoder)
            decoder = Conv1D(4, 1, padding='same', name='decoder_conv3')(decoder)
            decoder = Lambda(lambda x: K.softmax(x, axis=-1), name='output_softmax')(decoder)
            decoder = Lambda(lambda x: K.mean(K.reshape(x, (-1, self.n_sampler, self.window_size, self.n_channels)), axis=1), 
                name='output_mean')(decoder)
        elif name == 'mlp':
            if n_layers >= 2:
                decoder = Dense(128, activation='relu', name='decoder_dense2')(decoder)
            decoder = Dense(self.window_size*self.n_channels, name='decoder_dense3')(decoder)
            decoder = Lambda(lambda x: K.softmax(x, axis=-1), name='output_softmax')(decoder)
            decoder = Lambda(lambda x: K.mean(K.reshape(x, (-1, self.n_sampler, self.window_size, self.n_channels)), axis=1), 
                name='output_mean')(decoder)
        elif name == 'lstm':
            decoder = LSTM(64, name='encoder_lstm1', return_sequences=True)(decoder)
        return decoder 

    def __init__(self, window_size=256, n_channels=4, n_sampler=20, latent_size=2, n_layers=1, name='conv'):
        '''Encoders and decoders and multi-layer fully-connected networks.
        Input are motif instances.
        '''
        self.window_size = window_size
        self.n_channels = n_channels
        self.n_sampler = n_sampler
        self.latent_size = latent_size
        self.n_layers = n_layers

        input_layer = Input(shape=(window_size, n_channels), name='input')
        encoder = self.build_encoder(input_layer, name=name, n_layers=n_layers)

        latent_params = Dense(latent_size*2, name='latent_params')(encoder)
        latent_mean = Lambda(lambda x: x[:, :latent_size], name='latent_mean')(latent_params)
        latent_logstd = Lambda(lambda x: x[:, latent_size:], name='latent_logstdd')(latent_params)
        latent_std = Lambda(lambda x: K.exp(x), name='latent_std')(latent_logstd)
        latent_var = Lambda(lambda x: K.square(x), name='latent_var')(latent_std)
        standard_sampler = Lambda(lambda x: K.random_normal((K.shape(x)[0], n_sampler, latent_size)),
                                  name='standard_sampler')(input_layer)
        latent_sampler = Lambda(lambda x: x[0]*K.expand_dims(x[2], axis=1) + K.expand_dims(x[1], axis=1), 
                                name='latent_sampler')([standard_sampler, latent_mean, latent_std])
        latent = Lambda(lambda x: K.reshape(x, (-1, latent_size)), name='latent')(latent_sampler)

        decoder = self.build_decoder(latent, name=name, n_layers=n_layers)

        def kl_loss(y_true, y_pred):
            KL = 0.5*K.sum(latent_var + K.square(latent_mean) - 1 - K.log(latent_var), axis=1)
            return KL
        
        def nll_loss(y_true, y_pred):
            y_true = K.reshape(y_true, (-1, n_channels))
            y_pred = K.reshape(y_pred, (-1, n_channels))
            NLL = K.categorical_crossentropy(y_true, y_pred)
            NLL = K.sum(K.reshape(NLL, (-1, window_size)), axis=1)
            return NLL
            
        def vae_loss(y_true, y_pred):
            return nll_loss(y_true, y_pred) + kl_loss(y_true, y_pred)    

        ll_input = Input(shape=(window_size, n_channels), name='ll_input')
        ll = -nll_loss(ll_input, decoder)
        model = Model(inputs=[input_layer], outputs=[decoder])
        model.compile(loss=vae_loss,
                      metrics=['categorical_accuracy', kl_loss, nll_loss],
                      optimizer='Adam')
        
        self.ll_function = K.function([ll_input, latent_sampler], [ll])
        # returns multinomial probabilities
        self.sample_function = K.function([latent], [decoder])
        
        self.input_layer = input_layer
        self.model = model
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.latent_params = latent_params
        self.latent_sampler = latent_sampler
        self.decoder = decoder
    
    def logL(self, X, n_sampler=20, batch_size = 100):
        ll_values = []
        for X_batch in batch_generator(X, batch_size=batch_size):
            sampler_batch = np.random.normal(size=(X_batch.shape[0], n_sampler, self.latent_size))
            ll_values.append(self.ll_function([X_batch, sampler_batch])[0])
        return np.concatenate(ll_values)

    def sample(self, size=1, batch_size = 100):
        X = []
        for indices in batch_generator(np.arange(size), batch_size=batch_size):
            sampler_batch = np.random.normal(size=(indices.shape[0], self.latent_size))
            p = self.sample_function([sampler_batch])[0]
            X.append(random_multinomial(p))
        X = np.concatenate(X, axis=0)
        return X

model_functions = {cls.__name__:cls() for cls in ModelFunction.__subclasses__()}

def get_model(name):
    if name.endswith('.seq_only'):
        return model_functions[name.strip('.seq_only')]
    elif name.endswith('.observed_reactivity') or name.endswith('.predicted_reactivity'):
        return partial(model_functions[name.split('.')[0]], n_channels=5)
    else:
        return model_functions[name]