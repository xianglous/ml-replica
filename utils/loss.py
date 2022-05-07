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
