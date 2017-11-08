# -*- coding: utf-8 -*-

# Learning and inference of Regime-Switching Model based on the idea of
# hidden markov model
# Part of the code are from Python package 'hmmlearn'
# Their source code can be found here: https://github.com/hmmlearn/hmmlearn
# Author: Liuyi Hu

from __future__ import print_function

import sys
import string
from collections import deque

import numpy as np
from numpy.linalg import multi_dot
from numpy.linalg import inv

from scipy.stats import multivariate_normal

from sklearn.base import BaseEstimator, _pprint
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import KMeans

from .utils import normalize

class ConvergenceMonitor(object):
    """Monitors and reports convergence to :data:'sys.stderr'.


    Parameters
    ----------
    tol: double
        Convergence threshold. EM has converged either if the maximum
        number of iterations is reached or the log probability
        improvement between the two consecutive iterations is less than
        threshold

    n_iter: int
        Maximum number of iterations to perform

    verbose: bool
        If "True" then per-iteration convergence reports are printed,
        otherwise the monitor is mute

    Attributes
    ----------
    history: deque
        The log probability of the data for the last two training
        iterations. If the values are not strictly increasing, the
        model did not converge.

    iter: int
        Number of iterations performed while training the model
    """

    _template = "{iter:>10d} {logprob:>16.4f} {delta:>16.4f}"

    def __init__(self, tol, n_iter, verbose):
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque(maxlen=2)
        self.iter = 0

    def __repr__(self):
        class_name = self.__class__.__name__
        params = dict(vars(self), history=list(self.history))
        return "{0}({1})".format(class_name, _pprint(params, offset=len(class_name)))

    def report(self, logprob):
        """Reports convergence to :data:'sys.stderr'.

        The output consists of three columns: iteration number, log probability
        of the data at the current iteration and convergence rate. At the first
        iteration, convergence rate is unknown and is thus denoted by NaN.

        Parameters
        ----------
        logprob: float
            The log probability of the data as computed by EM algorithm in the
            current iteration
        """
        if self.verbose:
            delta = logprob - self.history[-1] if self.history else np.nan
            message = self._template.format(iter=self.iter+1, logprob=logprob, delta=delta)
            print(message, file=sys.stderr)

        self.history.append(logprob)
        self.iter += 1

    @property
    def converged(self):
        """``True`` if the EM algorithm converged and ``False`` otherwise
        """
        return (self.iter == self.n_iter or (len(self.history) == 2 and self.history[1] - self.history[0] >= 0 and self.history[1] - self.history[0] <= self.tol))



class HMMRS(BaseEstimator):
    """Hidden Markov Models with Regime-Switch Model

    Parameters
    ----------
    random_state: RandomState or an int seed, optional
        A random number generator instance.

    n_components: int
        Number of states in the model

    n_iter: int, optional
        Maximum number of iterations to perform

    tol: float, optional
        Convergence threshold for the EM algorithm

    vebose: bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, and other characters for
        subclass-specific emission parameters. Defaults to all
        parameters.

    Attributes
    ----------
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    loadingmat_: array, shape (n_components, # of stocks, # of risk factors + 1)
        Loading matrix B for the Regime-Switch model under different states (with intercept)

    covmat_: array, shape (n_components, # of stocks, # of stocks)
        Covariance matrix in the Regime-Switch model

    """
    def __init__(self, random_state=None, n_components=1, n_iter=1000, tol=1e-4, verbose=False, init_params=string.ascii_letters):
        self.random_state = random_state
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.init_params = init_params

    def _init(self, Y, F):
        """Initializes model parameters prior to fitting

        Parameters
        ----------
        Y: array-like, shape(n_samples, n_features)
            Stock price matrix of individual samples

        F: array-like, shape(n_samples, # of risk factors + 1))
            Risk factor matrix of individual samples
        """
        init = 1. / self.n_components
        if 's' in self.init_params or not hasattr(self, "startprob_"):
            self.startprob_ = np.full(self.n_components, init)
        if 't' in self.init_params or not hasattr(self, "transmat_"):
            self.transmat_ = np.full((self.n_components, self.n_components), init)
        # apply the K-means algorithm to initialize the mean and covariance
        kmeans = KMeans(n_clusters=self.n_components, random_state=0).fit(Y)
        if 'l' in self.init_params or not hasattr(self, "loadingmat_"):
            self.loadingmat_ = np.random.rand(self.n_components, Y.shape[1], F.shape[1])
            for component in range(self.n_components):
                self.loadingmat_[component,:,0] = np.mean(Y[kmeans.labels_ == component])
        if 'c' in self.init_params or not hasattr(self, "covmat_"):
            self.covmat_ = np.zeros((self.n_components, Y.shape[1], Y.shape[1]))
            for component in range(self.n_components):
                self.covmat_[component,:] = np.cov(Y[kmeans.labels_ == component], rowvar=False)

    def _check(self):
        """Validates model parameters prior to fitting.

        Raises
        ------

        ValueError
            If any of the parameters are invalid, e.g. if :attr:'startprob_' don't sum to 1.
        """
        self.startprob_ = np.asarray(self.startprob_)
        if len(self.startprob_) != self.n_components:
            raise ValueError("startprob_ must have length n_components")
        if not np.allclose(self.startprob_.sum(), 1.0):
            raise ValueError("startprob_ must sum to 1.0 (got {0:.4f})".format(self.startprob_.sum()))

        self.transmat_ = np.asarray(self.transmat_)
        if self.transmat_.shape != (self.n_components, self.n_components):
            raise ValueError("transmat_ must have shape (n_components, n_components)")
        if not np.allclose(self.transmat_.sum(axis=1), 1.0):
            raise ValueError("rows of transmat_ must sum to 1.0 (got {0})".format(self.transmat_.sum(axis=1)))

    def _compute_likelihood(self, Y, F, log=False):
        """Computes per-component probability under the mdoel
        P(Y_t∣h_t = i)

        Parameters
        ----------
        Y: array-like, shape(n_samples, n_features)
            Stock price matrix of individual samples

        F: array-like, shape(n_samples, n_features)
            Risk factor matrix of individual samples

        Returns
        -------
        prob: array, shape(n_samples, n_components)
            Pprobability of each sample in ``Y, T'' for each of the hidden states
        """
        n_samples, _ = Y.shape
        prob = np.zeros((n_samples, self.n_components))
        for t in xrange(n_samples):
            for state in range(self.n_components):
                m = np.dot(self.loadingmat_[state], F[t])
                if log:
                    prob[t, state] = multivariate_normal(mean=m, cov=self.covmat_[state]).logpdf(Y[t])
                else:
                    prob[t, state] = multivariate_normal(mean=m, cov=self.covmat_[state]).pdf(Y[t])
        return prob

    def _do_forward_pass(self, frameprob):
        n_samples, n_components = frameprob.shape
        logl = 0.0
        fwdlattice = np.zeros((n_samples, n_components))
        scaling_factor = np.zeros(n_samples)

        for state in range(n_components):
            fwdlattice[0, state] = frameprob[0, state] * self.startprob_[state]
        scaling_factor[0] = fwdlattice[0].sum()
        logl += np.log(scaling_factor[0])
        normalize(fwdlattice[0])

        for t in xrange(1, n_samples):
            for state in range(n_components):
                fwdlattice[t, state] = frameprob[t, state] * np.dot(fwdlattice[t-1], self.transmat_[:,state])
            scaling_factor[t] = fwdlattice[t].sum()
            logl += np.log(scaling_factor[t])
            normalize(fwdlattice[t])

        return fwdlattice, logl, scaling_factor

    def _do_backward_pass(self, frameprob, scaling_factor):
        n_samples, n_components = frameprob.shape
        bwdlattice = np.zeros((n_samples, n_components))
        bwdlattice[n_samples-1] = 1.0

        for t in xrange(n_samples-2, -1, -1):
            for state in range(n_components):
                temp = 0.0
                for l in range(n_components):
                    temp += bwdlattice[t+1, l] * frameprob[t+1, l] * self.transmat_[state, l]

                bwdlattice[t, state] = temp / scaling_factor[t+1]

        return bwdlattice

    def _compute_posterior(self, fwdlattice, bwdlattice, scaling_factor, frameprob):
        """Compute P(h_t = i ∣ Y_1, ..., Y_T) and
            P(h_t=i, h_t-1=j ∣ Y_1, ..., Y_T)
        """
        n_samples, n_components = frameprob.shape
        posterior_state = np.zeros((n_samples, n_components))
        posterior_transmat = np.zeros((n_samples, n_components, n_components))
        for t in xrange(n_samples):
            for i in range(n_components):
                posterior_state[t, i] = fwdlattice[t, i] * bwdlattice[t, i]
                if t > 0:
                    for j in range(n_components):
                        posterior_transmat[t, i, j] = bwdlattice[t, i] * frameprob[t, i] * self.transmat_[j, i] * fwdlattice[t-1, j] / scaling_factor[t]


        return posterior_state, posterior_transmat

    def _do_M_step(self, posterior_state, posterior_transmat, Y, F):
        n_samples, n_components = posterior_state.shape
        self.startprob_ = posterior_state[0] / sum(posterior_state[0])
        for i in range(n_components):
            for j in range(n_components):
                self.transmat_[j,i] = posterior_transmat[:, i, j].sum()
        normalize(self.transmat_, axis=1)
        for i in range(n_components):
            self.loadingmat_[i] = multi_dot([Y.T, np.diag(posterior_state[:,i]), F, inv(multi_dot([F.T, np.diag(posterior_state[:,i]), F]))])
            tmp = np.zeros((Y.shape[1], Y.shape[1]))
            for t in xrange(n_samples):
                v = np.atleast_2d(Y[t]).T - np.dot(self.loadingmat_[i], np.atleast_2d(F[t]).T)
                tmp += posterior_state[t,i] * np.dot(v, v.T)

            self.covmat_[i] = tmp / posterior_state[:,i].sum()

    def fit(self, Y, F):
        """Estimate model parameters

        Parameters
        ----------
        Y: array-like, shape(n_samples, n_features)
            Stock price matrix of individual samples

        F: array-like, shape(n_samples, n_features)
            Risk factor matrix of individual samples

        Returns
        -------
        self: object
            Returns self.
        """
        Y = check_array(Y)
        F = check_array(F)
        if Y.shape[0] != F.shape[0]:
            raise ValueError("rows of Y must match rows of F")
        self._init(Y, F)
        self._check()

        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)
        for iter in range(self.n_iter):
            frameprob = self._compute_likelihood(Y, F)
            # do forward pass
            fwdlattice, logl, scaling_factor = self._do_forward_pass(frameprob)
            # do backward pass
            bwdlattice = self._do_backward_pass(frameprob, scaling_factor)

            # calculate g_{i,t} and h_{ji,t}
            posterior_state, posterior_transmat = self._compute_posterior(fwdlattice, bwdlattice, scaling_factor, frameprob)
            # do M-step
            self._do_M_step(posterior_state, posterior_transmat, Y, F)
            self.monitor_.report(logl)
            if self.monitor_.converged:
                break

        return self



    def predict(self, Y, F):
        """Find most likely state sequence corresponding to ``Y, F``

        Parameters
        ----------
        Y: array, shape(n_samples, # of stocks)
        F: array, shape(n_samples, # of risk factors + 1)

        Returns
        -------
        state_sequence: array, shape(n_samples, )
            Labels for each sample from ``Y, F``

        logprob: float, the loglikeihood of P(Y_1, ...Y_T, h_1, ..., h_T)
                where the hidden state sequence is the most likely sequence found by Viterbi algorithm
        """
        framelogprob = self._compute_likelihood(Y, F, log=True)
        n_samples, n_components = framelogprob.shape
        state_sequence = np.empty(n_samples, dtype=np.int32)
        viterbi_lattice = np.full((n_samples, n_components), -np.inf)
        log_startprob = np.log(self.startprob_)
        log_transmat = np.log(self.transmat_)
        tmp = np.empty(n_components)

        for i in range(n_components):
            viterbi_lattice[0, i] = log_startprob[i] + framelogprob[0, i]
        # Induction
        for t in xrange(1, n_samples):
            for i in range(n_components):
                for j in range(n_components):
                    tmp[j] = viterbi_lattice[t-1, j] + log_transmat[j, i]
                viterbi_lattice[t, i] = np.amax(tmp) + framelogprob[t, i]
        # Backtracking
        state_sequence[n_samples-1] = idx = np.argmax(viterbi_lattice[n_samples-1])
        logprob = viterbi_lattice[n_samples-1, idx]
        for t in range(n_samples-2, -1, -1):
            for i in range(n_components):
                tmp[i] = viterbi_lattice[t, i] + log_transmat[i, idx]
            state_sequence[t] = idx = np.argmax(tmp)

        return np.asarray(state_sequence), logprob, viterbi_lattice

    def predict_streaming(self, Y, F, last_state):
        """Find most likely state when observations of ``Y, F`` becomes
        available one by one.

        Parameters
        ----------
        Y: array, shape(n_samples, # of stocks)
        F: array, shape(n_samples, # of risk factors + 1)

        Returns
        -------
        state_sequence: array, shape(n_samples, )
            Labels for each sample from ``Y, F``

        logprob: float, the loglikeihood of P(Y_1, ...Y_T, h_1, ..., h_T)
                where the hidden state sequence is the most likely sequence found by Viterbi algorithm
        """
        n_samples = Y.shape[0]
        log_transmat = np.log(self.transmat_)
        state_sequence = np.empty(n_samples, dtype=np.int32)
        for t in range(n_samples):
            y = np.atleast_2d(Y[t])
            f = np.atleast_2d(F[t])
            logprob = self._compute_likelihood(y, f, log=True)
            _, n_components = logprob.shape
            tmp = np.empty(n_components)

            for i in range(n_components):
                tmp[i] = log_transmat[last_state, i] + logprob[0, i]
            state_sequence[t] = last_state = np.argmax(tmp)
        return np.asarray(state_sequence)


    def sample(self, F, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        F: array, shape(n_samples, # of risk factors)

        random_state : RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.

        Returns
        -------
        Y : array, shape (n_samples, n_features)
            Feature matrix.

        state_sequence : array, shape (n_samples, )
            State sequence produced by the model.
        """
        check_is_fitted(self, "startprob_")
        self._check()

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        startprob_cdf = np.cumsum(self.startprob_)

        transmat_cdf = np.cumsum(self.transmat_, axis=1)
        #transmat_cdf = np.array([[0.9, 1.0], [0.7, 1.0]])
        currstate = (startprob_cdf > random_state.rand()).argmax()
        state_sequence = [currstate]
        Y = [self._generate_sample_from_state(currstate, F[0], random_state=random_state)]

        n_samples = F.shape[0]
        for t in range(n_samples-1):
            currstate = (transmat_cdf[currstate] > random_state.rand()).argmax()
            state_sequence.append(currstate)
            Y.append(self._generate_sample_from_state(currstate, F[t+1], random_state=random_state))

        return np.atleast_2d(Y), np.array(state_sequence, dtype=int)

    def _generate_sample_from_state(self, state, f, random_state=None):
        """Generates a random sample from a given component

        Parameters
        ----------
        state: int
            Index of the compont to condition on

        f: array, shape(# of risk factors + 1, )

        random_stae: RandomState or an int seed
            A random number generator instance. If ``None``, the object's ``random_state`` is used

        Returns
        -------
        y: array, shape(n_features, )
            A random sample form the emission distribution corresponding to a given componnt
        """
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        m = np.dot(self.loadingmat_[state], f)
        y = np.random.multivariate_normal(mean=m, cov=self.covmat_[state])

        return y
