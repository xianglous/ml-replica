import numpy as np


def hinge_loss(X, y, weights):
    """
    X: (n, m)
    y: (n, )
    weights: (m, )
    """
    return np.mean(np.maximum(0, 1 - y * (X @ weights)))


def MSE_loss(X, y, weights):
    """
    X: (n, m)
    y: (n, )
    weights: (m, )
    """
    return 0.5 * np.mean((y - X @ weights) ** 2)


def cross_entropy_loss(X, y, weights):
    """
    X: (n, m)
    y: (n, )
    weights: (m, )
    """
    return -np.mean(y * np.log(X @ weights) + (1 - y) * np.log(1 - X @ weights))


def logistic_loss(X, y, weights):
    """
    X: (n, m)
    y: (n, )
    weights: (m, )
    """
    return np.mean(np.log(1 + np.exp(-y * (X @ weights))))