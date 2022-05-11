import sys
sys.path.append('..')
import numpy as np
import time
from typing import Union, Type, Optional
from utils.data import Dataset
from utils.model import BaseModel
from utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix
from NonLinearClassification.decision_tree import DecisionTree
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import mode


def fit_single(estimator, X, y):
    return estimator.fit(X, y)


def predict_single(estimator, X):
    return estimator.predict(X)


class BaggingClassifier(BaseModel):

    def __init__(self, 
                 base_estimator:Optional[Type[BaseModel]]=None,
                 n_estimators:int=10, 
                 max_samples:int|float=1.0, 
                 max_features:int|float=1.0, 
                 bootstrap:bool=True, 
                 bootstrap_features:bool=False, 
                 n_jobs:int=1,
                 random_state:Optional[int]=None):
        if base_estimator is None:
            base_estimator = DecisionTree()
        self.base_estimator = base_estimator
        self.estimators = [self.base_estimator.clone() for _ in range(n_estimators)]
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.n_jobs = n_jobs
        if random_state is not None:
            np.random.seed(random_state)
    
    def __bootstrap_sample(self, X, y):
        if isinstance(self.max_samples, float):
            n_samples = int(len(X) * self.max_samples)
        else:
            n_samples = self.max_samples
        indices = np.random.choice(X.shape[0], n_samples, replace=self.bootstrap)
        return X[indices], y[indices]
    
    def __bootstrap_features(self, features):
        if isinstance(self.max_features, float):
            n_features = int(len(features) * self.max_features)
        else:
            n_features = self.max_features
        indices = np.random.choice(len(features), n_features, replace=self.bootstrap_features)
        return features[indices]

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.features = np.arange(X.shape[1])
        self.estimator_features = []
        works = []
        for estimator in self.estimators:
            X_, y_ = self.__bootstrap_sample(X, y)
            features = self.__bootstrap_features(self.features)
            X_ = X_[:, features]
            works.append((estimator, X_, y_))
            self.estimator_features.append(features)
        with ProcessPoolExecutor(max_workers=self.n_jobs) as exe:
            self.estimators = list(exe.map(fit_single, *zip(*works)))

    def predict(self, X):
        works = []
        for estimator, features in zip(self.estimators, self.estimator_features):
            X_ = X[:, features]
            works.append((estimator, X_))
        with ProcessPoolExecutor(max_workers=self.n_jobs) as exe:
            predictions = np.array(list(exe.map(predict_single, *zip(*works)))).T
        return mode(predictions, axis=1)[0].T


def evaluate(data, x_cols, y_col, max_depth=10, criterion="entropy", random_state=None):
    print("==========================")
    start = time.time()
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    base = DecisionTree(max_depth=max_depth, criterion=criterion)
    clf = BaggingClassifier(
        base_estimator=base, 
        n_estimators=5, 
        max_samples=0.8, 
        max_features=0.3, 
        bootstrap_features=True, 
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



if __name__ == '__main__':
    data = Dataset("../Data/drug200.csv", random_state=42)
    x_cols = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
    y_col = "Drug"
    evaluate(data, x_cols, y_col, max_depth=10, criterion="entropy", random_state=42)