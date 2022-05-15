import numpy as np
from typing import Optional, Type
from ..utils.model import BaseModel


class EnsembleModel(BaseModel):
    def __init__(self, 
            base_estimator:Type[BaseModel],
            n_estimators:int,
            n_jobs:int,
            random_state:Optional[int]=None):
        super().__init__(
            n_estimators=n_estimators, 
            n_jobs=n_jobs, 
            random_state=random_state)
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = [self.base_estimator.clone() for _ in range(n_estimators)]
        self.n_jobs = n_jobs
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
    
    def predict(self, X):
        super().predict(X)