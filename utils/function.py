import numpy as np


def sigmoid(X, weights):
    """
    X: (n, m)
    weights: (m, )
    """
    return 1 / (1 + np.exp(-X @ weights))