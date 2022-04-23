import numpy as np
import pandas as pd


def gradient_descent(X, y, lr=1e-3, max_iter=1000):
    """
    X: (n, m)
    y: (n, )
    """
    n, m = X.shape
    weights = np.zeros(m)
    num_iter = 0
    while num_iter < max_iter:
        weights = weights + lr * (y - (weights @ X.T)) @ X
        num_iter += 1
    return weights


def stochastic_gradient_descent(X, y, lr=1e-3, max_iter=1000):
    """
    X: (n, m)
    y: (n, )
    """
    n, m = X.shape
    weights = np.zeros(m)
    num_iter = 0
    while num_iter < max_iter:
        for i in range(n):
            weights = weights + lr * y[i] * X[i]
            num_iter += 1
    return weights