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


def logistic_loss(X, y, weights):
    """
    X: (n, m)
    y: (n, )
    weights: (m, )
    """
    return np.mean(np.log(1 + np.exp(-y * (X @ weights))))


def cross_entropy_loss(y, num_classes):
    """
    y: (n, )
    """
    loss = 0
    for i in range(num_classes):
        loss += np.mean(y == i) * np.log(np.mean(y == i))
    return -loss