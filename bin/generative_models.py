import numpy as np
import numba
import scipy.stats

class Parameter(np.ndarray):
    pass

def parameter(object, dtype=None, **kwargs):
    '''Create a Parameter object, same as np.array()
    '''
    a = np.array(object, dtype=dtype, **kwargs)
    return Parameter(a.shape, buffer=a, dtype=a.dtype)

class Parameter(np.ndarray):
    pass

def parameter(object, dtype=None, **kwargs):
    '''Create a Parameter object, same as np.array()
    '''
    a = np.array(object, dtype=dtype, **kwargs)
    return Parameter(a.shape, buffer=a, dtype=a.dtype)

class GenerativeModel(object):
    def logL(self, X):
        raise NotImplementedError('this method should be implemented in subclasses')

    def init_params(self):
        raise NotImplementedError('this method should be implemented in subclasses')

    def fit(self, X):
        raise NotImplementedError('this method should be implemented in subclasses')
    
    def get_params(self, as_dict=True):
        if as_dict:
            params = {key:val for key, val in self.__dict__.items() if isinstance(val, Parameter)}
        else:
            params = [val for key, val in self.__dict__.items() if isinstance(val, Parameter)]
        return params

    def set_params(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, Parameter(val))
    
    def sample(self, size, *args, **kwargs):
        raise NotImplementedError('this method should be implemented in subclasses')

class GaussianDistribution(GenerativeModel):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = parameter(np.array(mu))
        self.sigma = parameter(sigma)
    
    def init_params(self):
        self.mu = parameter(np.random.normal())
        self.sigma = parameter(np.random.gamma(1))
    
    def logL(self, X):
        return scipy.stats.norm.logpdf(X, loc=self.mu, scale=self.sigma)

    def fit(self, X, weights=None):
        if weights is None:
            mu, sigma = scipy.stats.norm.fit(X)
        else:
            N_w = np.sum(weights)
            mu = np.sum(X*weights)/N_w
            sigma = np.sqrt(np.sum(weights*np.square(X - mu))/N_w)
        self.mu = parameter(mu)
        self.sigma = parameter(sigma)
    
    def sample(self, size=3):
        return scipy.stats.norm.rvs(self.mu, self.sigma, size=size)
    
class BackgroundModel(GenerativeModel):
    def __init__(self, length=3, n_channels=4):
        self.length = length
        self.n_channels = n_channels
        self.p = parameter(np.full(n_channels, 1.0/n_channels))

    @numba.jit
    def logL(self, X):
        '''Compute log likelihood of input sequences
        Args:
            X: one-hot coding of sequences. ndarray of shape (n_seqs, length, n_channels)
        Returns:
            log likelihood. ndarray of shape (n_seqs,)
        '''
        logP = np.log(self.p)
        N, M, K = X.shape
        logL = np.zeros(N)
        for i in range(N):
            for j in range(M):
                for k in range(K):
                    logL[i] += X[i, j, k]*logP[k]
        return logL
    
    def init_params(self, alpha=10.0):
        prior = scipy.stats.dirichlet(np.full(self.n_channels, alpha))
        self.p = parameter(prior.rvs(size=1)[0])
    
    def fit(self, X, weights=None):
        '''Fit model parameters to input data
        Args:
            X: one-hot coding of sequences. ndarray of shape (n_seqs, length, n_channels)
            weights: per-sequence weights. ndarray of shape (n_seqs,)
        '''
        N, M, K = X.shape
        if weights is not None:
            X = X*weights[:, np.newaxis, np.newaxis]
            self.p = parameter(np.mean(np.sum(X, axis=0)/np.sum(weights), axis=0))
        else:
            self.p = parameter(np.mean(X.reshape((-1, K)), axis=0))
    
    def sample(self, size=3):
        '''Sample from the distribution
        Returns:
            one-hot coding of sequences. ndarray of shape (n_seqs, length, n_channels)
        '''
        X = np.random.choice(self.n_channels, size=(size, self.length), p=self.p)
        X = (X[:, :, np.newaxis] == np.arange(self.n_channels, dtype=np.int32)).astype(np.int32)
        return X

class PwmModel(GenerativeModel):
    def __init__(self, length=3, n_channels=4):
        self.length = length
        self.n_channels = n_channels
        self.pwm = parameter(np.full((length, n_channels), 1.0/n_channels))
    
    @numba.jit
    def logL(self, X):
        N, M, K = X.shape
        logP = np.log(self.pwm)
        logL = np.zeros(N)
        for i in range(N):
            for j in range(M):
                for k in range(K):
                    logL[i] += X[i, j, k]*logP[j, k]
        return logL

    def init_params(self, alpha=0.5):
        prior = scipy.stats.dirichlet(np.full(self.n_channels, alpha))
        self.pwm = parameter(prior.rvs(size=self.length))
    
    def fit(self, X, weights=None):
        '''Fit model parameters to input data
        Args:
            X: one-hot coding of sequences. ndarray of shape (n_seqs, length, n_channels)
            weights: per-sequence weights. ndarray of shape (n_seqs,)
        '''
        N, M, K = X.shape
        if weights is not None:
            X = X*weights[:, np.newaxis, np.newaxis]
            self.pwm = parameter(np.sum(X, axis=0)/np.sum(weights))
        else:
            self.pwm = parameter(np.mean(X, axis=0))
    
    def sample(self, size=3):
        '''Sample from the distribution
        Returns:
            one-hot coding of sequences. ndarray of shape (n_seqs, length, n_channels)
        '''
        X = np.empty((size, self.length), dtype=np.int32)
        for i in range(self.length):
            X[:, i] = np.random.choice(self.n_channels, size=size, p=self.pwm[i])
        X = (X[:, :, np.newaxis] == np.arange(self.n_channels, dtype=np.int32)).astype(np.int32)
        return X

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