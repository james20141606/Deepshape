import numpy as np
import h5py
from generative_models import GenerativeModel, parameter
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

class Encoder(object):
    def __init__(self, n_latent=2):
        self.n_latent = n_latent
    
    def __call__(self, x):
        '''Build an encoder network
        Args:
            x: input tensor of shape [batch_size] + input_shape
        Returns:
            tensor of shape [batch_size, n_latent*2]
        '''
        raise NotImplementedError('this function must be implemented by subclasses')
    
class SimpleEncoder(Encoder):
    def __call__(self, x):
        if len(x.shape) > 2:
            x = Flatten(name='encoder_flatten')(x)
        x = Dense(self.n_latent*2, name='encoder_dense')(x)
        return x

class MLPEncoder(Encoder):
    def __init__(self, n_latent=2, layers=None):
        super(MLPEncoder, self).__init__(n_latent)
        if layers is not None:
            self.layers = layers
        else:
            self.layers = []

    def add_layer(self, n_hidden):
        self.layers.append(n_hidden)

    def __call__(self, x):
        if len(x.shape) > 2:
            x = Flatten(name='encoder_flatten')(x)
        for i, n_hidden in enumerate(self.layers):
            x = Dense(n_hidden, name='encoder_dense{}'.format(i + 1), activation='relu')(x)
        x = Dense(self.n_latent*2, name='encoder_output', activation='relu')(x)
        return x

class Decoder(object):
    def __init__(self, output_shape, activation=None):
        if not (isinstance(output_shape, list) or isinstance(output_shape, tuple)):
            output_shape = (output_shape)
        self.output_shape = output_shape
        self.activation = activation
    
    def __call__(self, x):
        '''Build a decoder network
        Args:
            x: input tensor of shape [batch_size, n_latent]
        Returns:
            tensor of shape [batch_size] + output_shape
        '''
        raise NotImplementedError('this function must be implemented by subclasses')

class SimpleDecoder(Decoder):
    def __call__(self, x):
        x = Dense(np.prod(self.output_shape), name='decoder_dense')(x)
        if len(self.output_shape) > 1:
            x = Reshape(self.output_shape, name='output_reshape')(x)
        if self.activation is not None:
            x = Activation(self.activation, name='output_activation')(x)
        return x

class SequenceDecoder(Decoder):
    def __call__(self, x):
        length, n_channels = self.output_shape
        x = Dense(length*n_channels, name='decoder_dense')(x)
        x = Reshape((length, n_channels), name='output_reshape')(x)
        x = Lambda(lambda x: K.softmax(x, axis=-1), name='output_activation')(x)
        return x

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

class VariationalAutoencoder(GenerativeModel):
    def __init__(self, input_shape, encoder, decoder, 
            n_latent=2, n_sampler=10, 
            likelihood='categorical',
            batch_size=25, epochs=10):
        if not (isinstance(input_shape, list) or isinstance(input_shape, tuple)):
            input_shape = [input_shape]
        else:
            input_shape = list(input_shape)
        self.input_shape = input_shape
        self.n_latent = n_latent
        self.encoder = encoder
        self.decoder = decoder
        self.n_sampler = n_sampler
        self.likelihood = likelihood
        self.batch_size =  batch_size
        self.epochs = epochs
        self.build()
    
    def build(self):
        # build encoder network
        input_layer = Input(shape=self.input_shape, name='input')
        encoder_output = self.encoder(input_layer)
        # build latent network
        latent_params = Dense(self.n_latent*2, name='latent_params')(encoder_output)
        logvar = Lambda(lambda x: K.clip(x[:, :self.n_latent], -5, 5), name='logvar')(latent_params)
        mu = Lambda(lambda x: x[:, self.n_latent:], name='mu')(latent_params)
        var = Lambda(lambda x: K.exp(x), name='var')(logvar)
        std = Lambda(lambda x: K.sqrt(x), name='std')(var)
        gaussian_sampler = Lambda(lambda x: K.random_normal((K.shape(x)[0], self.n_sampler, self.n_latent)),
            name='gaussian_sampler')(input_layer)
        latent_sampler = Lambda(lambda x: x[0]*K.expand_dims(x[2], axis=1) + K.expand_dims(x[1], axis=1),
            name='latent_sampler')([gaussian_sampler, mu, std])
        latent_values = Lambda(lambda x: K.reshape(x, (-1, self.n_latent)), name='latent')(latent_sampler)
        # build decoder network
        decoder_output = self.decoder(latent_values)
        output = Lambda(lambda x: K.mean(K.reshape(x, [-1, self.n_sampler] + self.input_shape), axis=1),
            name='output_mean')(decoder_output)
        # define loss functions
        def kl_loss(y_true, y_pred):
            KL = 0.5*K.sum(var + K.square(mu) - 1 - K.log(var), axis=1)
            return KL
            
        def sequence_nll_loss(y_true, y_pred):
            y_shape = K.shape(y_true)
            y_true = K.reshape(y_true, (-1, y_shape[-1]))
            y_pred = K.reshape(y_pred, (-1, y_shape[-1]))
            NLL = K.categorical_crossentropy(y_true, y_pred)
            NLL = K.sum(K.reshape(NLL, (-1, y_shape[1])), axis=1)
            return NLL
        
        def sequence_accuracy(y_true, y_pred):
            return K.mean(K.cast(K.equal(K.argmax(y_true, axis=-1),
                        K.argmax(y_pred, axis=-1)),
                        K.floatx()), axis=1)
        
        def sequence_vae_loss(y_true, y_pred):
            return sequence_nll_loss(y_true, y_pred) + kl_loss(y_true, y_pred)

        def nll_loss(y_true, y_pred):
            NLL = K.categorical_crossentropy(y_true, y_pred)
            return NLL
                
        #def vae_loss(y_true, y_pred):
        #    return nll_loss(y_true, y_pred) + kl_loss(y_true, y_pred)

        import likelihoods
        ll = getattr(likelihoods, self.likelihood)
        def vae_loss(y_true, y_pred):
            return -ll(y_true, y_pred) + kl_loss(y_true, y_pred)
        
        # build training model
        model = Model(inputs=[input_layer], outputs=[output])
        model.compile(loss=vae_loss,
            metrics=[sequence_accuracy, kl_loss, nll_loss],
            optimizer='Adam')
        self.model = model
        # build log likelihood function
        ll_input = Input(shape=self.input_shape, name='ll_input')
        ll_output = ll(ll_input, output)
        self.ll_function = K.function([ll_input, latent_sampler], [ll_output])
        # build function for generating new samples
        self.sampler_function = K.function([ll_input, latent_values], [output])
    
    def logL(self, X):
        ll_values = []
        for X_batch in batch_generator(X, batch_size=self.batch_size):
            sampler_batch = np.random.normal(size=(X_batch.shape[0], self.n_sampler, self.n_latent))
            ll_values.append(self.ll_function([X_batch, sampler_batch])[0])
        return np.concatenate(ll_values)
    
    def sample(self, size=10):
        X = []
        for indices in batch_generator(np.arange(size), batch_size=self.batch_size):
            sampler_batch = np.random.normal(size=(indices.shape[0], self.n_latent))
            p = self.sample_function([sampler_batch])[0]
            X.append(random_multinomial(p))
        X = np.concatenate(X, axis=0)
        return X
    
    def fit(self, X, weights=None, verbose=0):
        self.model.fit(X, X, batch_size=self.batch_size, epochs=self.epochs, sample_weight=weights, verbose=verbose)
    
    def init_params(self):
        session = K.get_session()
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

class MixtureDensityVAE(GenerativeModel):
    def __init__(self, input_shape, encoder, decoders, 
            n_latent=2, n_sampler=10, 
            likelihood='categorical',
            batch_size=25, epochs=10):
        self.input_shape = input_shape
        self.encoder = encoder,
        self.decoders = decoders,
        self.n_latent = n_latent
        self.build()

    def init_params(self):
        session = K.get_session()
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
                
    def build(self):
        pass
        
class MotifVariationalAutoencoder(GenerativeModel):
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
        