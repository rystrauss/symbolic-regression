import numpy as np


def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r_squared(y_true, y_pred):
    numerator = np.square(y_true, y_pred).sum()
    denominator = np.square(y_true, np.mean(y_true)).sum()
    return 1 - (numerator / denominator)

