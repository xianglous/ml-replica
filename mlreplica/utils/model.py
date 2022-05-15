import numpy as np
from ..utils.metrics import accuracy, r_squared


class BaseModel:
    """Base class for all models"""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, X:np.ndarray):
        return self.predict(X)
    
    def fit(self, X:np.ndarray, y:np.ndarray, sample_weight:np.ndarray=None):
        pass

    def predict(self, X:np.ndarray):
        pass

    def str(self):
        """Return a string representation of the model"""
        return self.__class__.__name__
    
    def __str__(self):
        return self.str()

    def clone(self):
        """Return a clone of the model"""
        return self.__class__(*self.args, **self.kwargs)


class BaseClassifier:
    """Base class for all classifiers"""
    def score(self, X:np.ndarray, y:np.ndarray, sample_weight:np.ndarray=None):
        return accuracy(y, self.predict(X), sample_weight)


class BaseRegressor:
    """Base class for all regressors"""
    def score(self, X:np.ndarray, y:np.ndarray, sample_weight:np.ndarray=None):
        return r_squared(y, self.predict(X), sample_weight)