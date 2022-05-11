from utils.algorithm import SGD, GD

class BaseModel:
    """
    Base class for all models
    """
    def __init__(self, **kwargs):
        self.params = kwargs
    
    def clone(self):
        return self.__class__(**self.params)

    
class LinearModel(BaseModel):
    """
    Linear model
    """
    def __init__(self):
        super().__init__()
        self.weights = None
    
    def fit(self, X, y, method='SGD', **kwargs):
        """
        X: (n, m)
        y: (n, )
        method: 'SGD' or 'GD'
        """
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
        if self.weights is None:
            raise ValueError("model is not fitted yet")
        return X @ self.weights