import sys

from regex import D
sys.path.append('..')
import numpy as np
from utils.data import prepare_data
from utils.metrics import accuracy, precision, recall, f1_score
from utils.loss import cross_entropy_loss


class decision_tree:

    class node:
        def __init__(self, feature, threshold, left=None, right=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            

    def __init__(self, max_depth=None, criterion="entropy"):
        self.max_depth = max_depth
        self.loss_func = cross_entropy_loss
    
    def __best_split(self, X, y, feature):
        threshold = None
        max_loss = 0
        for val in set(X[:, feature]):
            left_idx = X[:, feature] <= val
            right_idx = X[:, feature] > val
            loss = np.mean(left_idx) * self.loss_func(y[left_idx]) + \
                     np.mean(right_idx) * self.loss_func(y[right_idx])
            if loss > max_loss:
                max_loss = loss
                threshold = val
        return threshold, max_loss

    def __best_feature(self, X, y):
        max_loss = 0
        for feature in self.features:
            threshold, loss = self.__best_split(X, y, feature)
            if loss > max_loss:
                max_loss = loss
                best_feature = feature
        return best_feature, threshold

    def __train(self, X, y, depth=0):
        if len(self.features) == 0 or depth == self.max_depth:
            return self.node(None, None, None)
        if len(set(y)) == 1:
            return self.node(None, None, None)
        best_feature, threshold = self.__best_feature(X, y)
        self.features.remove(best_feature)
        left_idx = X[:, best_feature] <= threshold
        right_idx = X[:, best_feature] > threshold
        left = self.__train(X[left_idx], y[left_idx], depth+1)
        right = self.__train(X[right_idx], y[right_idx], depth+1)
        return self.node(best_feature, threshold, left, right)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.features = set(range(X_train.shape[1]))
        self.root = self.__train(X_train, y_train)
    
    def __predict(self, node, x):
        if x[node.feature] <= node.threshold:
            if node.left is None:
                return 0
            return self.__predict(node.left, x)
        else:
            if node.right is None:
                return 1
            return self.__predict(node.right, x)

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            y_pred.append(self.__predict(self.root, x))
        return np.array(y_pred)
        
