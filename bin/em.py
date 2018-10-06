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

class MixtureModel(GenerativeModel):
    def __init__(self, models, p_mix=None):
        self.models = models
        self.n_components = len(models)
        if p_mix is not None:
            self.p_mix = parameter(p_mix)
        else:
            self.init_params()
    
    def init_params(self, components=False):
        self.p_mix = parameter(np.full(self.n_components, 1.0/self.n_components))
        if components:
            for m in self.models:
                m.init_params()

    def get_params(self, as_dict=True):
        if as_dict:
            params = {}
            params['p_mix'] = self.p_mix
            for i, m in enumerate(self.models):
                params['model[%d]'%i] = m.get_params(as_dict)
        return params

    def logL(self, X):
        logP = np.concatenate([m.logL(X).reshape((-1, 1)) for m in self.models], axis=1)
        logL = np.log(np.sum(np.exp(logP)*self.p_mix[np.newaxis, :], axis=1))
        return logL

    def fit_step(self, X, return_internal=False):
        # E-step
        logL_c = np.concatenate([m.logL(X).reshape((-1, 1)) for m in self.models], axis=1)
        L_c = np.exp(logL_c)
        weights = L_c*self.p_mix[np.newaxis, :]
        weights /= np.sum(weights, axis=1, keepdims=True)
        #print('L_c: ({}, {})'.format(np.mean(L_c), np.std(L_c)))
        #print('weights: ({}, {})'.format(np.mean(weights), np.std(weights)))
        #print(self.get_params())
        N_weighted = np.sum(weights, axis=0)
        #weights /= N_weighted[np.newaxis, :]
        # M-step
        for i, m in enumerate(self.models):
            m.fit(X, weights=weights[:, i])
        self.p_mix = N_weighted/np.sum(N_weighted)
        if return_internal:
            return {'weights': weights, 'L_c': L_c, 'logL_c': logL_c, 'params': self.get_params()}
        else:
            return {}
    
    def fit(self, X, max_iter=30, n_runs=10, tol=1e-3, return_internal=False):
        best_logL = None
        best_results = None
        for i in range(n_runs):
            self.init_params(components=True)
            results = self.fit_round(X, max_iter=max_iter, tol=tol, return_internal=return_internal)
            if best_logL is None:
                best_logL = results[-1]['logL']
                best_results = results
            else:
                if results[-1]['logL'] > best_logL:
                    best_logL = results[-1]['logL']
                    best_results = results
        return best_results
        
    def fit_round(self, X, max_iter=30, tol=1e-3, return_internal=False):
        '''Train the model
        Args:
            X: input sequences, ndarray of shape (n_seqs, length, n_channels)
        '''
        results = []
        logL = np.sum(self.logL(X))
        logL_old = logL
        i_iter = 0
        while True:
            result = self.fit_step(X, return_internal)
            logL = np.sum(self.logL(X))
            result['logL'] = logL
            results.append(result)
            #print('iter = {}, logL = {}'.format(i_iter, logL))
            i_iter += 1
            if i_iter >= max_iter:
                break
            if abs(logL - logL_old) < tol:
                break
            logL_old = logL
        print('optimized in {} iterations, logL = {}'.format(i_iter, logL))
        return results

    def sample(self, size=3):
        components = np.random.choice(self.n_components, size=size, p=self.p_mix)
        X_c = np.concatenate([np.expand_dims(m.sample(size=size), axis=1) for m in self.models], axis=1)
        X = X_c[np.r_[:size], components]
        return X
    
    def posteriors(self, X):
        logL_c = np.concatenate([m.logL(X).reshape((-1, 1)) for m in self.models], axis=1)
        L_c = np.exp(logL_c)
        posteriors = L_c*self.p_mix[np.newaxis, :]
        posteriors /= np.sum(posteriors, axis=1, keepdims=True)
        return posteriors
    
    def predict(self, X):
        posteriors = self.posteriors(X)
        return np.argmax(posteriors, axis=1)


