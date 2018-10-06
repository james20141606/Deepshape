import numpy as np
import numba
import scipy.stats

from generative_models import GenerativeModel, parameter

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


