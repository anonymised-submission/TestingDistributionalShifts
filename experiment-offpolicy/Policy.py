from scipy.special import softmax
import numpy as np
import statsmodels.api as sm


def vectorized(prob_matrix, items):
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    return items[k]


class LinearPolicy(object):

    def __init__(self, reg_model, temp, bias=True, name=None):
        self.name = name
        self.bias = bias
        self.reg_model = reg_model
        self.n_actions = self.reg_model.shape[0]
        self.temp = temp

    def get_prob(self, X, A=None):
        if self.bias:
            X = sm.tools.add_constant(X)
        p = softmax((1 / self.temp) * self.reg_model.dot(X.T), axis=0)
        if A is None:
            return p
        else:
            return p[A, np.arange(len(A))]

    def get_actions(self, X):
        p = self.get_prob(X)
        A = vectorized(p, np.arange(self.n_actions))
        return A


class RandomPolicy(object):

    def __init__(self, n_actions, weights=None):
        self.n_actions = n_actions
        self.reg_model = np.ones(n_actions)[:, np.newaxis]
        if weights is None:
            self.weights = np.ones(shape=(self.n_actions,)) / n_actions
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()

    def get_prob(self, X, A=None):
        p = np.vstack([self.weights] * X.shape[0])
        if A is None:
            return p
        else:
            return p[np.arange(len(A)), A]

    def get_actions(self, X):
        A = np.random.choice(self.n_actions, X.shape[0], p=self.weights)
        return A
