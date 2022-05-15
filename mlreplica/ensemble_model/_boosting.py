import numpy as np
from ._base import EnsembleModel
from ..tree_model import DecisionTree


class AdaBoostClassifier(EnsembleModel):

    def __init__(self,
            base_estimator=None,
            n_estimators=50,
            learning_rate=1.0,
            random_state=None):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if base_estimator is None:
            base_estimator = DecisionTree(max_depth=3, splitter='best', random_state=random_state)
        super().__init__(base_estimator, n_estimators, 1, random_state)
        self.learning_rate = learning_rate
    
    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.sample_weight = np.ones(X.shape[0]) / X.shape[0]
        self.estimator_weights = np.zeros(self.n_estimators)
        for i in range(self.n_estimators):
            self.estimators[i].fit(X, y, sample_weight=self.sample_weight)
            y_pred = self.estimators[i].predict(X)
            error = np.sum(self.sample_weight[y != y_pred])
            if error == 0:
                break
            self.estimator_weights[i] = self.learning_rate * \
                (np.log((1 - error) / error) + np.log(self.num_classes - 1))
            self.sample_weight *= np.exp(self.sample_weight * (y != y_pred))
            sample_weight_sum = np.sum(self.sample_weight)
            if sample_weight_sum == 0:
                break
            self.sample_weight /= np.sum(self.sample_weight)
        return self

    def predict(self, X):
        preds = np.array([
            self.estimators[i].predict(X)
            if self.estimator_weights[i] > 0
            else np.zeros(X.shape[0])
            for i in range(self.n_estimators)]).T
        return np.round(np.average(preds, axis=1, weights=self.estimator_weights))
        
        
