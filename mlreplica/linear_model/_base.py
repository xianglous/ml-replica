from typing import Callable
import numpy as np
from ..utils.model import BaseModel


class LinearModel(BaseModel):
    """Linear model: y = X @ w + b (b can be omitted by adding a column of 1s to X)"""
    def __init__(self, 
            bias:bool, 
            solver:Callable, 
            transform:Callable,
            **kwargs):
        super().__init__(bias, solver, transform, **kwargs)
        self.weights = None
        self.bias = bias
        self.solver = solver # algorithm to solve the linear system
        self.transform = transform # transform applied to the prediction
    
    def fit(self, X, y, sample_weight=None):
        """
        X: (n, m)
        y: (n, )
        """
        super().fit(X, y, sample_weight)
        if self.bias:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        if self.solver is None:
            self.weights = np.zeros(X.shape[1])
        else:
            self.weights = self.solver(X, y, **self.kwargs)
        return self
    
    def predict(self, X):
        """
        X: (n, m)
        """
        super().predict(X)
        if self.weights is None:
            raise ValueError("model is not fitted yet")
        if self.bias:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        if self.transform is not None:
            return self.transform(X @ self.weights)
        return X @ self.weights

    def str(self, x_cols=None, y_col=None):
        if x_cols is None:
            x_cols = [f"x_{i}" for i in range(1, self.weights.shape[0] + 1 - self.bias)]
            y_col = "y"
        x_str = f"{'w_0 + ' if self.bias else ''}" +\
            f"{' + '.join([f'w_{i+1} * {x_cols[i]}' for i in range(len(x_cols))])}"
        if self.transform is not None:
            x_str = f"{self.transform.__name__}({x_str})"
        return f"Model: {y_col} = {x_str}"
