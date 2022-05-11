from locale import normalize
import sys
sys.path.append('..')
import numpy as np
import time
from collections.abc import Callable
from typing import Union, Type, Optional
from utils.data import Dataset, model_str
from utils.model import BaseModel
from utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix
from utils.loss import cross_entropy_loss, gini_index_loss


class DecisionTree(BaseModel):

    class treeNode: # tree node
        def __init__(self, y, feature, threshold, left=None, right=None):
            values, counts = np.unique(y, return_counts=True)
            self.pred = values[np.argmax(counts)] # mode of y
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            
    def __init__(self, max_depth:int=None,
            criterion:str|Callable[[np.ndarray, Optional[int]], float]="entropy"):
        super().__init__()
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
            threshold = (vals[i] + prev) / 2 # midpoint between adjacent values
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
        for feature in self.remained_features:
            threshold, loss = self.__best_split(X, y, feature)
            if loss < min_loss:
                min_loss = loss
                best_feature = feature
                best_threshold = threshold
        return best_feature, best_threshold, min_loss

    def __train(self, X, y, depth=0):
        if len(y) == 0: # no data left
            return None
        if len(self.remained_features) == 0 or\
                depth == self.max_depth: # no more features or max depth reached
            return self.treeNode(y, None, None)
        if len(set(y)) == 1: # all same class
            return self.treeNode(y, None, None)
        cur_loss = self.loss_func(y, self.num_classes) # current loss
        best_feature, threshold, loss = self.__best_feature(X, y)
        if loss == 0 or loss == cur_loss: # perfect split or no improve
            self.remained_features.remove(best_feature)
        left_idx = X[:, best_feature] <= threshold
        right_idx = X[:, best_feature] > threshold
        left = self.__train(X[left_idx], y[left_idx], depth+1)
        right = self.__train(X[right_idx], y[right_idx], depth+1)
        return self.treeNode(y, best_feature, threshold, left, right)

    def fit(self, X_train, y_train):
        super().fit(X_train, y_train) # check args
        self.X_train = X_train
        self.y_train = y_train
        self.remained_features = set(range(X_train.shape[1]))
        self.num_classes = len(set(y_train))
        self.root = self.__train(X_train, y_train)
        return self
    
    def __predict(self, node, x):
        if node.threshold is None:
            return node.pred # leaf node
        if x[node.feature] <= node.threshold:
            if node.left is None:
                return node.pred
            return self.__predict(node.left, x)
        else:
            if node.right is None:
                return node.pred
            return self.__predict(node.right, x)

    def predict(self, X_test):
        super().predict(X_test)
        y_pred = []
        for x in X_test:
            y_pred.append(self.__predict(self.root, x))
        return np.array(y_pred)

    def __tree_to_str(self, node, dataset, x_cols, y_col, middle_padding=3):
        if node is None: # end
            return [""], 0, 0
        if node.feature is None: # leaf
            pred = node.pred if dataset is None else dataset.translate(y_col, node.pred)
            root_str = "pred: {}".format(pred)
        else: # split
            feat = node.feature if dataset is None else x_cols[node.feature]
            root_str = f"{feat} <= {node.threshold}"
        idx = (len(root_str) - 1) // 2 # index of the root center
        if node.left is None and node.right is None: # leaf
            return [root_str], len(root_str), idx
        left_strs, left_width, left_idx = \
            self.__tree_to_str(node.left, dataset, x_cols, y_col, middle_padding) # left subtree
        right_strs, right_width, right_idx = \
            self.__tree_to_str(node.right, dataset, x_cols, y_col, middle_padding) # right subtree
        root_left_padding = max(0, left_idx - idx) # root left padding
        left_left_padding = max(0, idx - left_idx) # left subtree padding
        root_right_padding = 0 # root right padding
        sub_width = left_left_padding + left_width + right_width
        idx += root_left_padding
        if len(root_str) < sub_width + middle_padding:
            root_right_padding = \
                sub_width + middle_padding - len(root_str) - root_left_padding
        else:
            middle_padding = len(root_str) - sub_width
        width = sub_width + middle_padding
        # root
        lines = [f"{' ' * root_left_padding}{root_str}{' ' * root_right_padding}"]
        # connector
        left = ' ' * (idx - 1)
        left += ' ' if left_width == 0 else '|'
        right = ' ' if right_width == 0 else '\\' +\
            '_' * max(0, (right_idx + left_left_padding + 
                          left_width + middle_padding - len(left) - 2))            
        right += ' ' * (width - len(right) - len(left))
        lines.append(f"{left}{right}")
        # subtrees
        for i in range(max(len(left_strs), len(right_strs))):
            line = " " * left_left_padding
            line += left_strs[i] if i < len(left_strs) \
                else " " * left_width # left subtree
            line += " " * middle_padding # middle padding
            line += right_strs[i] if i < len(right_strs) \
                else " " * right_width # right subtree
            lines.append(line)
        return lines, width, idx

    def print_tree(self, dataset:Dataset=None, x_cols:list[str]=None, y_col:str=None):
        tree_lines, _, _ = self.__tree_to_str(self.root, dataset, x_cols, y_col)
        for line in tree_lines:
            print(line)


def evaluate(data, x_cols, y_col, max_depth=10, criterion="entropy"):
    print("==========================")
    start = time.time()
    X_train, y_train, X_test, y_test = data.get_split(x_cols, y_col)
    clf = DecisionTree(max_depth, criterion)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_acc = accuracy(y_train, y_train_pred)
    test_acc = accuracy(y_test, y_test_pred)
    print(f"{model_str(x_cols, y_col)} using {criterion}")
    print("Training accuracy:", train_acc)
    print("Testing accuracy:", test_acc)
    print("Training precision:")
    for cls in sorted(set(y_train)):
        print(f"{data.translate(y_col, cls)}: {precision(y_train, y_train_pred, cls)}")
    print("Testing precision:")
    for cls in sorted(set(y_train)):
        print(f"{data.translate(y_col, cls)}: {precision(y_test, y_test_pred, cls)}")
    print("Training recall:")
    for cls in sorted(set(y_train)):
        print(f"{data.translate(y_col, cls)}: {recall(y_train, y_train_pred, cls)}")
    print("Testing recall:")
    for cls in sorted(set(y_train)):
        print(f"{data.translate(y_col, cls)}: {recall(y_test, y_test_pred, cls)}")
    print("Training F1:")
    for cls in sorted(set(y_train)):
        print(f"{data.translate(y_col, cls)}: {f1_score(y_train, y_train_pred, cls)}")
    print("Testing F1:")
    for cls in sorted(set(y_train)):
        print(f"{data.translate(y_col, cls)}: {f1_score(y_test, y_test_pred, cls)}")
    print("Training Confusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred, len(set(y_train))))
    print("Testing Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred, len(set(y_train))))
    print(f"Training used {time.time()-start} seconds")
    # clf.print_tree(data, x_cols, y_col) # print tree
    print("==========================")

        



if __name__ == "__main__":
    features = [["Age", "Sex"], ["Age", "Sex", "BP"], ["Age", "Sex", "BP", "Cholesterol"], ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]]
    y_col = "Drug"
    data = Dataset("../Data/drug200.csv", random_state=42)
    for x_cols in features:
        evaluate(data, x_cols, y_col, max_depth=10, criterion="entropy")
        evaluate(data, x_cols, y_col, max_depth=10, criterion="gini")