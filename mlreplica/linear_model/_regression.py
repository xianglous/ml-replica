import numpy as np
from typing import Type
from ._base import LinearModel
from ..utils.algorithm import SGD, GD
from ..utils.loss import Loss, MSELoss
from ..utils.model import BaseRegressor


def closed_form(X, y, reg):
    """
    X: (n, m)
    y: (n, )
    reg: float
    w = (X.T @ X + reg * I)^-1 @ X.T @ y
    """
    return np.linalg.pinv(X.T @ X + reg * np.eye(X.shape[1])) @ X.T @ y


class LinearRegression(LinearModel, BaseRegressor):

    def __init__(self,
            bias=True,
            loss:str|Type[Loss]='mse',
            regularization="l2",
            alpha=1.0,
            solver=SGD,            
            transform=None,
            lr=1e-3,
            tol=1e-3,
            max_iter=1000):
        if isinstance(loss, str):
            if loss == 'mse':
                loss = MSELoss()
            else:
                raise ValueError(f"loss {loss} not supported")
        if regularization is None:
            alpha = 0.0
        if isinstance(solver, str):
            if solver == 'SGD':
                solver = SGD
            elif solver == 'GD':
                solver = GD
            elif solver == 'closed-form':
                solver = lambda x, y, **kwargs:closed_form(x, y, alpha)
            else:
                raise ValueError('Unknown solver: {}'.format(solver))
        super().__init__(
            bias=bias, 
            solver=solver, 
            transform=transform, 
            loss=loss,
            regularization=regularization, 
            alpha=alpha, 
            lr=lr, 
            tol=tol, 
            max_iter=max_iter)


class RidgeRegression(LinearRegression):

    def __init__(self,
            bias=True,
            alpha=1.0,
            solver='SGD',
            lr=1e-3,
            tol=1e-3,
            max_iter=1000):
        super().__init__(
            bias=bias, 
            regularization="l2", 
            alpha=alpha, 
            solver=solver, 
            lr=lr, 
            tol=tol, 
            max_iter=max_iter)


class LassoRegression(LinearRegression):

    def __init__(self,
            bias=True,
            alpha=1.0,
            solver='SGD',
            lr=1e-3,
            tol=1e-3,
            max_iter=1000):
        super().__init__(
            bias=bias, 
            regularization="l1", 
            alpha=alpha, 
            solver=solver, 
            lr=lr, 
            tol=tol, 
            max_iter=max_iter)