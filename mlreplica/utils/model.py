import numpy as np


class BaseModel:
    """Base class for all models"""
    def __init__(self, **kwargs):
        self.params = kwargs

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
        return self.__class__(**self.params)
