import numpy as np


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


def GD(X, y, grad_func, loss_func, regularization=None, alpha=1.0, lr=1e-3, tol=1e-3, max_iter=1000):
    """
    X: (n, m)
    y: (n, )
    grad_func: (X, y, weights) -> grad
    loss_func: (X, y, weights) -> loss
    regularization: None, 'l1', 'l2'
    alpha: regularization strength
    lr: learning rate
    tol: tolerance
    max_iter: maximum number of iterations
    """
    n, m = X.shape
    weights = np.zeros(m)
    num_iter = 0
    loss, prev_loss = None, None
    reg_grad = None
    if regularization == 'l1':
        reg_grad = l1_grad
    elif regularization == 'l2':
        reg_grad = l2_grad
    elif regularization is not None:
        raise ValueError('regularization must be None or "l1" or "l2"')
    while num_iter < max_iter:
        grad = grad_func(X, y, weights) +\
            (alpha * reg_grad(weights)
             if regularization is not None else 0) # gradient of all samples
        weights = weights - lr * grad # gradient descent
        prev_loss = loss
        loss = loss_func(X, y, weights)
        if prev_loss and abs(loss-prev_loss) < tol:
            return weights
        num_iter += 1
    return weights


def SGD(X, y, grad_func, loss_func, regularization=None, alpha=1.0, lr=1e-3, tol=1e-3, max_iter=1000):
    """
    X: (n, m)
    y: (n, )
    grad_func: (X, y, weights) -> grad
    loss_func: (X, y, weights) -> loss
    regularization: None, 'l1', 'l2'
    alpha: regularization strength
    lr: learning rate
    tol: tolerance
    max_iter: maximum number of iterations
    """
    n, m = X.shape
    weights = np.zeros(m)
    num_iter = 0
    loss, prev_loss = None, None
    reg_grad = None
    if regularization == 'l1':
        reg_grad = l1_grad
    elif regularization == 'l2':
        reg_grad = l2_grad
    elif regularization is not None:
        raise ValueError('regularization must be None or "l1" or "l2"')
    while num_iter < max_iter:
        for i in range(n):
            grad = grad_func(X[i], y[i], weights) +\
                (alpha * reg_grad(weights)
                 if regularization is not None else 0) # gradient of one sample
            weights = weights - lr * grad # gradient descent
            prev_loss = loss
            loss = loss_func(X[i], y[i], weights)
            if prev_loss and abs(loss-prev_loss) < tol:
                return weights
            num_iter += 1
    return weights
