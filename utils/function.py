import numpy as np


def sigmoid(X, weights):
    """
    X: (n, m)
    weights: (m, )
    """
    return 1 / (1 + np.exp(-X @ weights))


def polynomial_kernel(u, v, gamma, degree, coef0):
    """
    u: (n, )
    v: (n, )
    gamma: float
    degree: int
    coef: float
    """
    return (gamma*u @ v + coef0) ** degree


def rbf_kernel(u, v, gamma):
    """
    u: (n, )
    v: (n, )
    gamma: float
    """
    return np.exp(-gamma * np.linalg.norm(u - v) ** 2)


def sigmoid_kernel(u, v, gamma, coef0):
    """
    u: (n, )
    v: (n, )
    gamma: float
    coef: float
    """
    return np.tanh(gamma * u @ v + coef0)