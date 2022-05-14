from ._base import LinearModel
from ._regression import LinearRegression, RidgeRegression, LassoRegression
from ._perceptron import Perceptron
from ._logistic import LogisticRegression
from ._svm import SVM


__all__ = [
    "LinearModel",
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "Perceptron",
    "LogisticRegression",
    "SVM",
]