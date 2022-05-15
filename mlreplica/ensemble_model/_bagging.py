import numpy as np
from typing import Union, Type, Optional
from ._base import EnsembleModel
from ..utils.model import BaseModel
from ..tree_model import DecisionTree
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import mode


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


class BaggingClassifier(EnsembleModel):

    def __init__(self, 
            base_estimator=None,
            n_estimators:int=10, 
            max_samples:int|float=1.0, 
            max_features:int|float=1.0, 
            bootstrap:bool=True, 
            bootstrap_features:bool=False, 
            n_jobs:int=1,
            random_state:Optional[int]=None):
        if base_estimator is None:
            base_estimator = DecisionTree(random_state=random_state)
        super().__init__(base_estimator, n_estimators, n_jobs, random_state)
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
    
    def __bootstrap_sample(self, X, y, sample_weight):
        if isinstance(self.max_samples, float):
            n_samples = int(len(X) * self.max_samples)
        else:
            n_samples = self.max_samples
        indices = np.random.choice(X.shape[0], n_samples, replace=self.bootstrap)
        if sample_weight is None:
            return X[indices], y[indices], None
        return X[indices], y[indices], sample_weight[indices]
    
    def __bootstrap_features(self, features):
        if isinstance(self.max_features, float):
            n_features = int(len(features) * self.max_features)
        else:
            n_features = self.max_features
        indices = np.random.choice(len(features), n_features, replace=self.bootstrap_features)
        return features[indices]

    def fit(self, X, y, sample_weight=None):
        self.features = np.arange(X.shape[1]) # feature indices
        self.estimator_features = [] # bootstraped feature indices
        works = []  # works for multiprocessing
        for estimator in self.estimators:
            X_, y_, sample_weight_ = self.__bootstrap_sample(X, y, sample_weight)
            features = self.__bootstrap_features(self.features)
            X_ = X_[:, features]
            works.append((estimator, X_, y_, sample_weight_))
            self.estimator_features.append(features)
        with ProcessPoolExecutor(max_workers=self.n_jobs) as exe:
            self.estimators = list(exe.map(fit_single, *zip(*works)))
        return self

    def predict(self, X):
        works = [] # works for multiprocessing
        for estimator, features in zip(self.estimators, self.estimator_features):
            X_ = X[:, features]
            works.append((estimator, X_))
        with ProcessPoolExecutor(max_workers=self.n_jobs) as exe:
            predictions = np.array(list(exe.map(predict_single, *zip(*works)))).T
        return mode(predictions, axis=1)[0].flatten() # majority vote