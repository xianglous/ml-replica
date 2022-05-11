import sys
sys.path.append('..')
import numpy as np
import time
from typing import Union, Type, Optional
from scipy.stats import mode
from NonLinearClassification.decision_tree import DecisionTree
from utils.data import Dataset
from utils.model import BaseModel
from utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix
from concurrent.futures import ProcessPoolExecutor


def fit_single(estimator, X, y):
    """
    X: (n, m)
    y: (n, )
    fit a single estimator
    """
    return estimator.fit(X, y)


def predict_single(estimator, X):
    """
    X: (n, m)
    predict using single estimator
    """
    return estimator.predict(X)


class RandomForest(BaseModel):
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
        self.estimators = []
        for _ in range(n_estimators):
            self.estimators.append(DecisionTree(
                criterion=criterion,
                splitter='random',
                max_depth=max_depth,
                max_features=max_features,
                random_state=random_state))
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        if random_state is not None:
            np.random.seed(random_state)
    
    def __bootstrap_sample(self, X, y):
        indices = np.random.choice(X.shape[0], self.n_samples, replace=self.bootstrap)
        return X[indices], y[indices]

    def fit(self, X, y):
        if isinstance(self.max_samples, float):
            self.n_samples = int(len(X) * self.max_samples)
        else:
            self.n_samples = self.max_samples
        works = []
        for estimator in self.estimators:
            X_, y_ = self.__bootstrap_sample(X, y) # sample data
            works.append((estimator, X_, y_))
        with ProcessPoolExecutor(max_workers=self.n_jobs) as exe:
            self.estimators = list(exe.map(fit_single, *zip(*works)))
        return self
    
    def predict(self, X):
        works = []
        for estimator in self.estimators:
            works.append((estimator, X))
        with ProcessPoolExecutor(max_workers=self.n_jobs) as exe:
            predictions = np.array(list(exe.map(predict_single, *zip(*works)))).T
        return mode(predictions, axis=1)[0].T # majority vote


def evaluate(data, x_cols, y_col, max_depth=10, criterion="entropy", random_state=None):
    print("==========================")
    start = time.time()
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    clf = RandomForest(
        n_estimators=5,
        criterion=criterion,
        max_depth=max_depth,
        max_samples=0.8, 
        max_features=0.3, 
        n_jobs=3,
        random_state=random_state)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    print("Training accuracy:", accuracy(y_train, y_train_pred))
    print("Testing accuracy:", accuracy(y_test, y_test_pred))
    print("Training Confusion matrix:")
    print(confusion_matrix(y_train, y_train_pred))
    print("Testing Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print(f"Training used {time.time()-start} seconds")
    print("==========================")
    return clf


if __name__ == '__main__':
    data = Dataset("../Data/drug200.csv", random_state=42)
    x_cols = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
    y_col = "Drug"
    evaluate(data, x_cols, y_col, max_depth=5, criterion="entropy", random_state=42)