import numpy as np
from typing import Union, Type, Optional
from scipy.stats import mode
from ._base import EnsembleModel
from ..tree_model import DecisionTree
from concurrent.futures import ProcessPoolExecutor


def fit_single(estimator, X, y, sample_weight):
    """
    X: (n, m)
    y: (n, )
    fit a single estimator
    """
    return estimator.fit(X, y, sample_weight)


def predict_single(estimator, X):
    """
    X: (n, m)
    predict using single estimator
    """
    return estimator.predict(X)


class RandomForest(EnsembleModel):
    """Random Forest Classifier"""
    def __init__(self,
                 n_estimators:int=10,
                 criterion:str='gini',
                 max_depth:int=None,
                 max_samples:int|float=1.0,
                 max_features:str|int|float='auto',
                 bootstrap:bool=True, 
                 n_jobs:int=1,
                 random_state:Optional[int]=None):
        if isinstance(max_features, str):
            if max_features == 'auto':
                max_features = 'sqrt'
            if max_features != 'sqrt' and max_features != 'log2':
                raise ValueError('max_features must be "sqrt" or "log2"')
        elif isinstance(max_features, float):
            if max_features < 0 or max_features > 1:
                raise ValueError('max_features must be between 0 and 1')
        super().__init__(
            DecisionTree(
                criterion=criterion,
                splitter='random',
                max_depth=max_depth,
                max_features=max_features,
                random_state=random_state), 
            n_estimators, n_jobs, random_state)
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
    
    def __bootstrap_sample(self, X, y, sample_weight=None):
        indices = np.random.choice(X.shape[0], self.n_samples, replace=self.bootstrap)
        if sample_weight is None:
            return X[indices], y[indices], None
        return X[indices], y[indices], sample_weight[indices]

    def fit(self, X, y, sample_weight=None):
        if isinstance(self.max_samples, float):
            self.n_samples = int(len(X) * self.max_samples)
        else:
            self.n_samples = self.max_samples
        works = []
        for estimator in self.estimators:
            X_, y_, sample_weight_ = self.__bootstrap_sample(X, y, sample_weight) # sample data
            works.append((estimator, X_, y_, sample_weight_))
        with ProcessPoolExecutor(max_workers=self.n_jobs) as exe:
            self.estimators = list(exe.map(fit_single, *zip(*works)))
        return self
    
    def predict(self, X):
        works = []
        for estimator in self.estimators:
            works.append((estimator, X))
        with ProcessPoolExecutor(max_workers=self.n_jobs) as exe:
            predictions = np.array(list(exe.map(predict_single, *zip(*works)))).T
        return mode(predictions, axis=1)[0].flatten() # majority vote