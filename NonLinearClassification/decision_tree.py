import sys
sys.path.append('..')
import numpy as np
import time
from scipy import stats
from utils.data import dataset
from utils.metrics import accuracy, precision, recall, f1_score
from utils.loss import cross_entropy_loss


class decision_tree:

    class tree_node:
        def __init__(self, y, feature, threshold, left=None, right=None):
            self.probs = {}
            for y_ in y:
                if y_ not in self.probs:
                    self.probs[y_] = 0
                self.probs[y_] += 1
            for y_ in self.probs:
                self.probs[y_] /= len(y)
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            

    def __init__(self, max_depth=None, criterion="entropy"):
        self.max_depth = max_depth
        if criterion == "entropy":
            self.loss_func = cross_entropy_loss
    
    def __best_split(self, X, y, feature):
        threshold = None
        min_loss = np.inf
        for val in set(X[:, feature]):
            left_idx = X[:, feature] <= val
            right_idx = X[:, feature] > val
            loss = np.mean(left_idx) * self.loss_func(y[left_idx], self.num_classes) + \
                     np.mean(right_idx) * self.loss_func(y[right_idx], self.num_classes)
            if loss < min_loss:
                min_loss = loss
                threshold = val
        return threshold, min_loss

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
        return best_feature, best_threshold

    def __train(self, X, y, depth=0):
        if len(y) == 0:
            return None
        if len(self.features) == 0 or depth == self.max_depth:
            return self.tree_node(y, None, None)
        if len(set(y)) == 1:
            return self.tree_node(y, None, None)
        best_feature, threshold = self.__best_feature(X, y)
        self.features.remove(best_feature)
        left_idx = X[:, best_feature] <= threshold
        right_idx = X[:, best_feature] > threshold
        left = self.__train(X[left_idx], y[left_idx], depth+1)
        right = self.__train(X[right_idx], y[right_idx], depth+1)
        return self.tree_node(y, best_feature, threshold, left, right)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.features = set(range(X_train.shape[1]))
        self.num_classes = len(set(y_train))
        self.root = self.__train(X_train, y_train)
    
    def __predict(self, node, x):
        if node.threshold is None:
            return max(node.probs.keys(), key=lambda k: node.probs[k])
        if x[node.feature] <= node.threshold:
            if node.left is None:
                return max(node.probs.keys(), key=lambda k: node.probs[k])
            return self.__predict(node.left, x)
        else:
            if node.right is None:
                return max(node.probs.keys(), key=lambda k: node.probs[k])
            return self.__predict(node.right, x)

    def predict(self, X_test):
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
    print("Training accuracy:", train_acc)
    print("Testing accuracy:", test_acc)
    print(f"Training used {time.time()-start} seconds")
    print("==========================")


if __name__ == "__main__":
    features = [["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]]
    y_col = "Drug"
    data = dataset("../Data/drug200.csv")
    for x_cols in features:
        evaluate(data, x_cols, y_col, max_depth=len(x_cols), criterion="entropy")