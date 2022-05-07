import numpy as np


def GD(X, y, grad_func, loss_func, lr=1e-3, tol=1e-3, max_iter=1000):
    """
    X: (n, m)
    y: (n, )
    """
    n, m = X.shape
    weights = np.zeros(m)
    num_iter = 0
    loss, prev_loss = None, None
    while num_iter < max_iter:
        grad = grad_func(X, y, weights)
        weights = weights - lr * grad
        prev_loss = loss
        loss = loss_func(X, y, weights)
        if prev_loss and abs(loss-prev_loss) < tol:
            return weights
        num_iter += 1
    return weights


def SGD(X, y, grad_func, loss_func, lr=1e-3, tol=1e-3, max_iter=1000):
    """
    X: (n, m)
    y: (n, )
    """
    n, m = X.shape
    weights = np.zeros(m)
    num_iter = 0
    loss, prev_loss = None, None
    while num_iter < max_iter:
        for i in range(n):
            grad = grad_func(X[i], y[i], weights)
            weights = weights - lr * grad
            prev_loss = loss
            loss = loss_func(X[i], y[i], weights)
            if prev_loss and abs(loss-prev_loss) < tol:
                return weights
            num_iter += 1
    return weights
