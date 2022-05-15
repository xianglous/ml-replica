import numpy as np
from .function import sigmoid


class Loss:
    def __init__(self):
        pass

    def __call__(
        self, 
        y_true:np.ndarray, 
        y_pred:np.ndarray):
        return np.average(self.loss(y_true, y_pred))
    
    def loss(self, y_true, y_pred):
        pass

    def stochastic_grad(
        self, 
        X_i:np.ndarray, 
        y_i:int|float|np.ndarray, 
        weights:np.ndarray):
        pass

    def grad(self, X, y, weights):
        return np.average([self.stochastic_grad(X[i], y[i], weights) for i in range(X.shape[0])], axis=0)


class HingeLoss(Loss):
    def loss(self, y_true, y_pred):
        return np.maximum(0, 1 - y_true * y_pred)

    def stochastic_grad(self, X_i, y_i, weights):
        if y_i * (weights @ X_i.T) < 1:
            return -y_i * X_i
        return np.zeros(len(weights))


class MSELoss(Loss):

    def loss(self, y_true, y_pred):
        return 0.5 * (y_true - y_pred) ** 2
    
    def stochastic_grad(self, X_i, y_i, weights):
        return X_i * (X_i @ weights - y_i)
    
    def grad(self, X, y, weights):
        return X.T @ (X @ weights - y) / X.shape[0]


class LogLoss(Loss):
    def loss(self, y_true, y_pred):
        return np.log(1 + np.exp(-y_true * y_pred))
    
    def stochastic_grad(self, X_i, y_i, weights):
        return X_i * (sigmoid(X_i @ weights) - y_i)

    def grad(self, X, y, weights):
        return X.T @ (sigmoid(X @ weights) - y) / X.shape[0]


def cross_entropy_loss(y, num_classes=None, sample_weight=None):
    """
    y: (n, )
    loss = -sum_i(yi * log(pi))
    """
    if num_classes is None:
        num_classes = len(np.unique(y))
    loss = 0
    if len(y) == 0:
        return loss
    for i in range(num_classes):
        rate = np.average(y == i, axis=0, weights=sample_weight)
        if rate == 0 or rate == 1:
            continue
        loss += rate * np.log(rate)
    return -loss


def gini_index_loss(y, num_classes=None, sample_weight=None):
    """
    y: (n, )
    loss = 1 - sum_i(pi * (1 - pi))
    """
    if num_classes is None:
        num_classes = len(np.unique(y))
    loss = 1
    if len(y) == 0:
        return loss
    for i in range(num_classes):
        rate = np.average(y == i, axis=0, weights=sample_weight)
        if rate == 0:
            continue
        loss -= rate ** 2
    return loss