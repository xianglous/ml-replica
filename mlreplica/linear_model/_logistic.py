from ._regression import LinearRegression
from ..utils.function import sigmoid
from ..utils.loss import LogLoss
from ..utils.model import BaseClassifier


class LogisticRegression(LinearRegression):
    def __init__(self, 
            bias=True, 
            regularization='l2', 
            alpha=1.0, 
            solver='SGD', 
            lr=1e-3, 
            tol=1e-3, 
            max_iter=1000):
        super().__init__(
            bias=bias, 
            loss=LogLoss(),
            regularization=regularization, 
            alpha=alpha, 
            solver=solver, 
            transform=sigmoid, 
            lr=lr, 
            tol=tol, 
            max_iter=max_iter)
    
    def predict(self, X):
        return self.predict_proba(X) > 0.5

    def predict_proba(self, X):
        return super().predict(X)
    
    def score(self, X, y):
        return BaseClassifier.score(self, X, y)
