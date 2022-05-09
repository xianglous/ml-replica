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


def cross_entropy_loss(y, num_classes=2):
    """
    y: (n, )
    """
    loss = 0
    if len(y) == 0:
        return loss
    for i in range(num_classes):
        rate = np.mean(y == i)
        if rate == 0 or rate == 1:
            continue
        loss += rate * np.log(rate)
    return -loss