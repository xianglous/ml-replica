import numpy as np
from .metrics import accuracy
from .loss import Loss


def perceptron(X, y, max_iter=1000):
    """
    X: (n, m)
    y: (n, )
    """
    n, m = X.shape
    num_iter = 0
    weights = np.zeros(m)
    acc = 0
    while acc < 1 and num_iter < max_iter:
        for i in range(n):
            if y[i] * (weights @ X[i].T) <= 0:
                weights = weights + y[i] * X[i]
                num_iter += 1
        acc = accuracy(y, np.sign(weights @ X.T))
    return weights


def l1_grad(weights):
    """
    weights: (m, )
    grad = sgn(weights)
    """
    return np.sign(weights)


def l2_grad(weights):
    """
    weights: (m, )
    grad = weights
    """
    return weights


def GD(
    X:np.ndarray, 
    y:np.ndarray, 
    loss:Loss, 
    regularization:str=None, 
    alpha:int|float=1.0, 
    lr:int|float=1e-3, 
    tol:int|float=1e-3, 
    max_iter:int=1000):
    """
    X: (n, m)
    y: (n, )
    loss: Loss
    regularization: None, 'l1', 'l2'
    alpha: regularization strength
    lr: learning rate
    tol: tolerance
    max_iter: maximum number of iterations
    """
    n, m = X.shape
    weights = np.zeros(m)
    num_iter = 0
    los, prev_los = None, None
    reg_grad = None
    if regularization == 'l1':
        reg_grad = l1_grad
    elif regularization == 'l2':
        reg_grad = l2_grad
    elif regularization is not None:
        raise ValueError('regularization must be None or "l1" or "l2"')
    while num_iter < max_iter:
        grad = loss.grad(X, y, weights) +\
            (alpha * reg_grad(weights)
             if regularization is not None else 0) # gradient of all samples
        weights = weights - lr * grad # gradient descent
        prev_los = los
        los = loss(y, X @ weights)
        if prev_los and abs(los-prev_los) < tol:
            return weights
        num_iter += 1
    return weights


def SGD(
    X:np.ndarray, 
    y:np.ndarray, 
    loss:Loss, 
    regularization:str=None, 
    alpha:int|float=1.0, 
    lr:int|float=1e-3, 
    tol:int|float=1e-3, 
    max_iter:int=1000):
    """
    X: (n, m)
    y: (n, )
    loss: Loss
    regularization: None, 'l1', 'l2'
    alpha: regularization strength
    lr: learning rate
    tol: tolerance
    max_iter: maximum number of iterations
    """
    n, m = X.shape
    weights = np.zeros(m)
    num_iter = 0
    los, prev_los = None, None
    reg_grad = None
    if regularization == 'l1':
        reg_grad = l1_grad
    elif regularization == 'l2':
        reg_grad = l2_grad
    elif regularization is not None:
        raise ValueError('regularization must be None or "l1" or "l2"')
    while num_iter < max_iter:
        for i in range(n):
            grad = loss.stochastic_grad(X[i], y[i], weights) +\
                (alpha * reg_grad(weights)
                 if regularization is not None else 0) # gradient of one sample
            weights = weights - lr * grad # gradient descent
            prev_los = los
            los = loss(y[i], X[i] @ weights)
            if prev_los and abs(los-prev_los) < tol:
                return weights
            num_iter += 1
    return weights
