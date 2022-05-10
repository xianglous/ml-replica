import sys
sys.path.append('..')
import numpy as np
import time
from collections.abc import Callable
from typing import Union, Type, Optional
from utils.data import dataset, model_str
from utils.metrics import accuracy, precision, recall, f1_score
from utils.loss import cross_entropy_loss, gini_index_loss


class decision_tree:

    class tree_node: # tree node
        def __init__(self, y, feature, threshold, left=None, right=None):
            values, counts = np.unique(y, return_counts=True)
            self.pred = values[np.argmax(counts)]
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            
    def __init__(self, max_depth:int=None,
            criterion:str|Callable[[np.ndarray, Optional[int]], float]="entropy"):
        self.max_depth = max_depth
        if callable(criterion):
            self.loss_func = criterion
        elif criterion == "entropy":
            self.loss_func = cross_entropy_loss
        elif criterion == "gini":
            self.loss_func = gini_index_loss
        else:
            raise ValueError("criterion must be a callable or 'entropy' or 'gini'")
    
    def __best_split(self, X, y, feature):
        vals = sorted(set(X[:, feature]))
        prev = vals[0] - 1
        best_threshold = vals[0] - 0.5
        min_loss = np.inf
        for i in range(len(vals)):
            threshold = (vals[i] + prev) / 2
            prev = vals[i]
            left_idx = X[:, feature] <= threshold
            right_idx = X[:, feature] > threshold
            loss = np.mean(left_idx) * self.loss_func(y[left_idx], self.num_classes) + \
                     np.mean(right_idx) * self.loss_func(y[right_idx], self.num_classes)
            if loss < min_loss:
                min_loss = loss
                best_threshold = threshold
        return best_threshold, min_loss

    def __best_feature(self, X, y):
        min_loss = np.inf
        best_feature = None
        best_threshold = None
        for feature in self.features:
            threshold, loss = self.__best_split(X, y, feature)
            if loss < min_loss:
                min_loss = loss
                best_feature = feature
                best_threshold = threshold
        return best_feature, best_threshold, min_loss

    def __train(self, X, y, depth=0):
        if len(y) == 0:
            return None
        if len(self.features) == 0 or depth == self.max_depth:
            return self.tree_node(y, None, None)
        if len(set(y)) == 1:
            return self.tree_node(y, None, None)
        best_feature, threshold, loss = self.__best_feature(X, y)
        if loss == 0: # perfect split
            self.features.remove(best_feature)
        left_idx = X[:, best_feature] <= threshold
        right_idx = X[:, best_feature] > threshold
        left = self.__train(X[left_idx], y[left_idx], depth+1)
        right = self.__train(X[right_idx], y[right_idx], depth+1)
        return self.tree_node(y, best_feature, threshold, left, right)

    def fit(self, X_train:np.ndarray, y_train:np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.features = set(range(X_train.shape[1]))
        self.num_classes = len(set(y_train))
        self.root = self.__train(X_train, y_train)
    
    def __predict(self, node, x):
        if node.threshold is None:
            return node.pred
        if x[node.feature] <= node.threshold:
            if node.left is None:
                return node.pred
            return self.__predict(node.left, x)
        else:
            if node.right is None:
                return node.pred
            return self.__predict(node.right, x)

    def predict(self, X_test:np.ndarray):
        y_pred = []
        for x in X_test:
            y_pred.append(self.__predict(self.root, x))
        return np.array(y_pred)
        

def evaluate(data, x_cols, y_col, max_depth=10, criterion="entropy"):
    print("==========================")
    start = time.time()
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    clf = decision_tree(max_depth, criterion)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_acc = accuracy(y_train, y_train_pred)
    test_acc = accuracy(y_test, y_test_pred)
    print(f"{model_str(x_cols, y_col)} using {criterion}")
    print("Training accuracy:", train_acc)
    print("Testing accuracy:", test_acc)
    print(f"Training used {time.time()-start} seconds")
    print("==========================")


if __name__ == "__main__":
    features = [["Age", "Sex"], ["Age", "Sex", "BP"], ["Age", "Sex", "BP", "Cholesterol"], ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]]
    y_col = "Drug"
    data = dataset("../Data/drug200.csv")
    for x_cols in features:
        evaluate(data, x_cols, y_col, max_depth=10, criterion="entropy")
        evaluate(data, x_cols, y_col, max_depth=10, criterion="gini")