import numpy as np
from .function import sigmoid


def hinge_loss(y_true, y_pred):
    """
    X: (n, m)
    y: (n, )
    weights: (m, )
    """
    return np.mean(np.maximum(0, 1 - y_true * y_pred))


def hinge_loss_grad(X, y, weights):
    """
    X: (n, m)
    y: (n, )
    weights: (m, )
    grad = -y * X
    """
    conds = (y * (weights @ X.T) < 1)
    return -y[conds] @ X[conds] / (1 if len(X.shape)==1 else X.shape[0])


def MSE_loss(y_true, y_pred):
    """
    X: (n, m)
    y: (n, )
    weights: (m, )
    """
    return 0.5 * np.mean((y_true - y_pred) ** 2)


def MSE_loss_grad(X, y, weights):
    """
    X: (n, m) or (n, )
    y: (n, ) or scalar
    weights: (m, )
    grad = -X (y - y_pred)
    """
    if len(X.shape) == 1:
        return -X * (y - X @ weights)
    return -X.T @ (y - X @ weights) / X.shape[0]


def logistic_loss(y_true, y_pred):
    """
    X: (n, m)
    y: (n, )
    weights: (m, )
    """
    return np.mean(np.log(1 + np.exp(-y_true * y_pred)))


def logistic_loss_grad(X, y, weights):
    """
    X: (n, m)
    y: (n, )
    weights: (m, )
    grad = -y * X / (1 + exp(y * (weights @ X.T)))
    """
    if len(X.shape) == 1:
        return X * (sigmoid(X @ weights) - y)
    return X.T @ (sigmoid(X @ weights) - y) / X.shape[0]


def cross_entropy_loss(y, num_classes=None):
    """
    y: (n, )
    loss = -sum_i(yi * log(pi))
    """
    if num_classes is None:
        num_classes = np.max(y) + 1
    loss = 0
    if len(y) == 0:
        return loss
    for i in range(num_classes):
        rate = np.mean(y == i)
        if rate == 0 or rate == 1:
            continue
        loss += rate * np.log(rate)
    return -loss


def gini_index_loss(y, num_classes=None):
    """
    y: (n, )
    loss = 1 - sum_i(pi * (1 - pi))
    """
    if num_classes is None:
        num_classes = np.max(y) + 1
    loss = 1
    if len(y) == 0:
        return loss
    for i in range(num_classes):
        rate = np.mean(y == i)
        if rate == 0:
            continue
        loss -= rate ** 2
    return loss