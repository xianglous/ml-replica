import numpy as np

def accuracy(y_true, y_pred, sample_weight=None):
    """
    y_true: (n, )
    y_pred: (n, )
    """
    return np.average(y_true == y_pred, axis=0, weights=sample_weight)


def r_squared(y_true, y_pred, sample_weight=None):
    """
    y_true: (n, )
    y_pred: (n, )
    """
    y_true_mean = np.average(y_true, axis=0, weights=sample_weight)
    rss = np.average((y_true - y_pred)**2, axis=0, weights=sample_weight)
    tss = np.average((y_true - y_true_mean)**2, axis=0, weights=sample_weight)
    if tss == 0:
        return 0
    return 1 - rss / tss


def precision(y_true, y_pred, pos_class=1):
    """
    y_true: (n, )
    y_pred: (n, )
    prec = TP / (TP + FP)
    """
    true_pred = (y_pred == pos_class)
    true_positives = np.sum((y_true == y_pred)*true_pred)
    false_positives = np.sum((y_true != y_pred)*true_pred)
    if true_positives + false_positives == 0:
        return 0
    return true_positives / (true_positives + false_positives)


def recall(y_true, y_pred, pos_class=1):
    """
    y_true: (n, )
    y_pred: (n, )
    recall = TP / (TP + FN)
    """
    true_true = (y_true == pos_class)
    true_positives = np.sum((y_true == y_pred)*true_true)
    false_negatives = np.sum((y_true != y_pred)*true_true)
    if true_positives + false_negatives == 0:
        return 0
    return true_positives / (true_positives + false_negatives)


def f1_score(y_true, y_pred, pos_class=1):
    """
    y_true: (n, )
    y_pred: (n, )
    f1 = 2 * precision * recall / (precision + recall)
    """
    precision_ = precision(y_true, y_pred, pos_class)
    recall_ = recall(y_true, y_pred, pos_class)
    if precision_ + recall_ == 0:
        return 0
    return 2 * precision_ * recall_ / (precision_ + recall_)


def confusion_matrix(y_true, y_pred, num_classes=None, normalize=None):
    """
    y_true: (n, )
    y_pred: (n, )
    conf_mat[i, j] = number of times y_true == i and y_pred == j
    """
    if num_classes is None:
        num_classes = np.max(y_true) + 1
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = np.sum((y_true == i) * (y_pred == j))
            if normalize == "true":
                if np.sum(y_true == i) != 0:
                    confusion_matrix[i, j] /= np.sum(y_true == i)
            elif normalize == "pred":
                if np.sum(y_pred == j) != 0:
                    confusion_matrix[i, j] /= np.sum(y_pred == j)
            elif normalize == "all":
                if len(y_true) != 0:
                    confusion_matrix[i, j] /= len(y_true)
    return confusion_matrix