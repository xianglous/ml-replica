import numpy as np

def accuracy(y_true, y_pred):
    """
    y_true: (n, )
    y_pred: (n, )
    """
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, pos_class=1):
    """
    y_true: (n, )
    y_pred: (n, )
    """
    true_pred = (y_pred == pos_class)
    true_positives = np.sum((y_true == y_pred)*true_pred)
    false_positives = np.sum((y_true != y_pred)*true_pred)
    return true_positives / (true_positives + false_positives)


def recall(y_true, y_pred, pos_class=1):
    """
    y_true: (n, )
    y_pred: (n, )
    """
    true_true = (y_true == pos_class)
    true_positives = np.sum((y_true == y_pred)*true_true)
    false_negatives = np.sum((y_true != y_pred)*true_true)
    return true_positives / (true_positives + false_negatives)


def f1_score(y_true, y_pred, pos_class=1):
    """
    y_true: (n, )
    y_pred: (n, )
    """
    precision_ = precision(y_true, y_pred, pos_class)
    recall_ = recall(y_true, y_pred, pos_class)
    return 2 * precision_ * recall_ / (precision_ + recall_)