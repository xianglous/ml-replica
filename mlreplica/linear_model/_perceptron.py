import numpy as np
from ._base import LinearModel
from ..utils.algorithm import perceptron
from ..utils.model import BaseClassifier


class Perceptron(LinearModel, BaseClassifier):
    def __init__(self, bias=True, max_iter=1000):
        super().__init__(
            bias=bias, 
            solver=perceptron, 
            transform=np.sign, 
            max_iter=max_iter)