import numpy as np
from collections.abc import Callable
from typing import Union, Type, Optional
from ..utils.data import Dataset
from ..utils.model import BaseModel, BaseClassifier
from ..utils.loss import cross_entropy_loss, gini_index_loss


class DecisionTreeClassifier(BaseModel, BaseClassifier):
    """Decision Tree Classifier"""

    class treeNode:
        """Tree node"""
        def __init__(self, y, feature, threshold, left=None, right=None):
            values, counts = np.unique(y, return_counts=True)
            self.pred = values[np.argmax(counts)] # mode of y
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            
    def __init__(self,
                 criterion:str|Callable[[np.ndarray, Optional[int]], float]="entropy",
                 splitter:str="best", # best or random
                 max_depth:int=None, # max depth of tree, None for no limit
                 max_features:int|str|float=None, # max features to consider, None for all
                 random_state:int=None):
        if callable(criterion):
            self.loss_func = criterion
        elif criterion == "entropy":
            self.loss_func = cross_entropy_loss
        elif criterion == "gini":
            self.loss_func = gini_index_loss
        else:
            raise ValueError("criterion must be a callable or 'entropy' or 'gini'")
        self.max_depth = max_depth
        if splitter != "best" and splitter != "random":
            raise ValueError("splitter must be 'best' or 'random'")
        self.splitter = splitter
        if isinstance(max_features, str):
            if max_features == "auto":
                max_features = 'sqrt'
            if max_features != 'sqrt' and max_features != 'log2':
                raise ValueError("max_features must be 'sqrt' or 'log2' or 'auto'")
        elif isinstance(max_features, float):
            if max_features < 0 or max_features > 1:
                raise ValueError('max_features must be between 0 and 1')
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            max_features=max_features,
            random_state=random_state)
        self.max_features = max_features
        if random_state is not None:
            np.random.seed(random_state) 
    
    def __split(self, X, y, sample_weight, feature, cur_loss):
        vals = sorted(set(X[:, feature]))
        vals.append(vals[-1] + 1) # split at last value
        vals.insert(0, vals[0] - 1) # split at first value
        split_importance = np.zeros(len(vals) - 1) # importance of each split
        split_threshold = np.zeros(len(vals) - 1) # threshold of each split
        for i in range(1, len(vals)):
            threshold = (vals[i] + vals[i - 1]) / 2 # split at midpoint
            left_idx = X[:, feature] <= threshold
            right_idx = X[:, feature] > threshold
            loss = np.mean(left_idx) * \
                self.loss_func(y[left_idx], self.num_classes, sample_weight[left_idx]) + \
                    np.mean(right_idx) * \
                        self.loss_func(y[right_idx], self.num_classes, sample_weight[right_idx])
            split_importance[i - 1] = max(0, cur_loss - loss) # gain
            split_threshold[i - 1] = threshold
        total_importance = np.sum(split_importance)
        if total_importance == 0:
            return vals[0], 0
        if self.splitter == "best": # best split
            best_index = np.argmax(split_importance)
        else: # random split
            best_index = np.random.choice(
                len(split_importance),
                p=split_importance/np.sum(split_importance))
        return split_threshold[best_index], split_importance[best_index]
    
    def __best_feature(self, X, y, sample_weight, cur_loss):
        max_importance = 0
        best_feature = None
        best_threshold = None
        # bootstrap features
        feature_indices = np.random.choice(X.shape[1], self.n_features, replace=False)
        for feature in feature_indices:
            threshold, importance = self.__split(X, y, sample_weight, feature, cur_loss)
            if importance > max_importance:
                max_importance = importance
                best_feature = feature
                best_threshold = threshold
        return best_feature, best_threshold, max_importance

    def __train(self, X, y, sample_weight, depth=0):
        if len(y) == 0: # no data left
            return None
        if len(set(y)) == 1: # all same class or max depth reached
            return self.treeNode(y, None, None)
        if self.max_depth is not None and depth == self.max_depth: # max depth reached
            return self.treeNode(y, None, None)
        cur_loss = self.loss_func(y, self.num_classes, sample_weight) # current loss
        best_feature, threshold, importance = self.__best_feature(X, y, sample_weight, cur_loss)
        if importance == 0: # no better split
            return self.treeNode(y, None, None)
        left_idx = X[:, best_feature] <= threshold
        right_idx = X[:, best_feature] > threshold
        left = self.__train(X[left_idx], y[left_idx], sample_weight[left_idx], depth+1)
        right = self.__train(X[right_idx], y[right_idx], sample_weight[right_idx], depth+1)
        return self.treeNode(y, best_feature, threshold, left, right)

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight) # check args
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        if self.max_features is None:
            self.n_features = X.shape[1]
        if isinstance(self.max_features, float):
            self.n_features = int(X.shape[1] * self.max_features)
        elif isinstance(self.max_features, int):
            self.n_features = self.max_features
        elif self.max_features == 'sqrt':
            self.n_features = int(np.sqrt(X.shape[1]))
        else:
            self.n_features = int(np.log2(X.shape[1]))
        self.num_classes = len(set(y))
        self.root = self.__train(X, y, sample_weight=sample_weight)
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

    def __tree_to_str(self, node, dataset, x_cols, y_col):
        if node is None: # end
            return []
        if node.feature is None: # leaf
            pred = node.pred if dataset is None else dataset.translate(y_col, node.pred)
            root_str = f"pred: {pred}"
        else: # split
            feat = node.feature if dataset is None else x_cols[node.feature]
            thresh = round(node.threshold, 2)
            root_str = f"{feat} <= {thresh}"
        idx = (len(root_str) - 1) // 2 # index of the root center
        if node.left is None and node.right is None: # leaf
            return [(root_str, 0, len(root_str))]
        left_strs = \
            self.__tree_to_str(node.left, dataset, x_cols, y_col) # left subtree
        right_strs = \
            self.__tree_to_str(node.right, dataset, x_cols, y_col) # right subtree
        left_idx = left_strs[0][1] + (left_strs[0][2] - 1) // 2 if left_strs else 0
        right_idx = right_strs[0][1] + (right_strs[0][2] - 1) // 2 if right_strs else 0
        root_left_padding = max(0, left_idx - idx) # root left padding
        left_left_padding = max(0, idx - left_idx) # left subtree padding
        min_dis = 0
        for i in range((min(len(left_strs), len(right_strs)) + 1) // 2):
            index = 2 * i
            if index >= len(right_strs):
                break
            right_start = right_strs[index][1]
            dis = right_start - len(left_strs[index][0]) - left_left_padding
            min_dis = min(min_dis, dis)
        mid_padding = -min_dis
        mid_padding += 2
        idx += root_left_padding
        # root
        first_line = f"{' ' * root_left_padding}{root_str}"
        lines = [(first_line, root_left_padding, len(root_str))]
        # connector
        left = ' ' * (idx - 1)
        left += ' ' if len(left_strs) == 0 else '|'
        right_indent = right_strs[0][1]
        right = ' ' if len(right_strs) == 0 else '\\' +\
            '_' * max(0, (right_idx + mid_padding + right_indent - len(left) - 2))            
        second_line = f"{left}{right}"
        lines.append((second_line, 0, len(second_line)))
        # subtrees
        for i in range(max(len(left_strs), len(right_strs))):
            line = " " * left_left_padding
            left_len = 0
            if i < len(left_strs):
                line += left_strs[i][0]
                start = left_left_padding + left_strs[i][1]
                left_len = len(left_strs[i][0])
            else:
                start = mid_padding + right_strs[i][1]
            if i < len(right_strs):
                line += " " * (mid_padding + right_strs[i][1] - left_len - left_left_padding) # middle padding
                line += right_strs[i][0][right_strs[i][1]:] # right subtree
            length = len(line) - start
            lines.append((line, start, length))
        return lines

    def str(self, dataset:Dataset=None, x_cols:list[str]=None, y_col:str=None):
        tree_lines = self.__tree_to_str(self.root, dataset, x_cols, y_col)
        tree_str = "\n"
        for line in tree_lines:
            tree_str += line[0] + "\n"
        return tree_str + "\n"