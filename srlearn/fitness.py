"""Contains various metrics for evaluating individual programs, i.e. fitness functions.

Author: Ryan Strauss
Author: Sarah Hancock
"""
import numpy as np


class _Fitness:
    """A thin wrapper class around the fitness functions."""

    def __init__(self, function, maximize, max_value):
        """Constructor.

        Args:
            function (callable): A fitness function that has the arguments (y_true, y_pred).
            maximize (bool): True iff the fitness function should be maximized.
            max_value (int): If maximize is True, this is the maximal (best) value that the fitness function can return.
        """
        self.function = function
        self.maximize = maximize
        self.max_value = max_value

    def __call__(self, *args):
        return self.function(*args)


def _mean_squared_error(y_true, y_pred):
    """Calculates mean squared error.

    Args:
        y_true (ndarray of shape (n_samples)): Ground truth target values.
        y_pred (ndarray of shape (n_samples)): Estimated target values.

    Returns:
        A non-negative floating point value (the best value is 0.0).
    """
    return np.mean(np.square(y_true - y_pred))


def _mean_absolute_error(y_true, y_pred):
    """Calculates mean absolute error.

        Args:
            y_true (ndarray of shape (n_samples)): Ground truth target values.
            y_pred (ndarray of shape (n_samples)): Estimated target values.

        Returns:
            A non-negative floating point value (the best value is 0.0).
        """
    return np.mean(np.abs(y_true - y_pred))


def _r_squared(y_true, y_pred):
    """R^2 (coefficient of determination) regression score.

    Best possible score is 1.0 and it can be negative (because the model can be arbitrarily bad).
    A constant model that always predicts the expected value of y, disregarding the input features,
    would get an R^2 score of 0.0.

    Args:
        y_true (ndarray of shape (n_samples)): Ground truth target values.
        y_pred (ndarray of shape (n_samples)): Estimated target values.

    Returns:
        The R^2 score.
    """
    numerator = np.square(y_true - y_pred).sum()
    denominator = np.square(y_true - np.mean(y_true)).sum()
    return 1 - (numerator / denominator)


mean_squared_error = _Fitness(_mean_squared_error, False, None)
mean_absolute_error = _Fitness(_mean_absolute_error, False, None)
r_squared = _Fitness(_r_squared, True, 1)

__all__ = [mean_squared_error, mean_absolute_error, r_squared]
