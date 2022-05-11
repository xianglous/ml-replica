import numpy as np
from utils.algorithm import SGD, GD

class BaseModel:
    """Base class for all models"""
    def __init__(self, **kwargs):
        self.params = kwargs
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        pass

    def predict(self, X:np.ndarray):
        pass

    def clone(self):
        """Return a clone of the model"""
        return self.__class__(**self.params)

    
class LinearModel(BaseModel):
    """Linear model"""
    def __init__(self):
        super().__init__()
        self.weights = None
    
    def fit(self, X, y, method='SGD', **kwargs):
        """
        X: (n, m)
        y: (n, )
        method: 'SGD' or 'GD'
        """
        super().fit(X, y)
        if callable(method):
            self.weights = method(X, y, **kwargs)
        elif method == 'SGD':
            self.weights = SGD(X, y, **kwargs)
        elif method == 'GD':
            self.weights = GD(X, y, **kwargs)
        else:
            raise ValueError("method must be 'SGD', 'GD', or a callable")
        return self
    
    def predict(self, X):
        """
        X: (n, m)
        """
        super().predict(X)
        if self.weights is None:
            raise ValueError("model is not fitted yet")
        return X @ self.weights