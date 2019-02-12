"""Defines the functions available for inclusion in the genetic algorithm's function set.

Functions defined here have the closure property, which requires that each of the functions in the
function set be able to accept, as its arguments, any value and data type that may possibly be
returned by any function in the function set and any value and data type that may possibly be
assumed by any terminal in the terminal set. [1]

[1] J. R. Koza, Genetic programming: on the programming of computers by means of natural selection. 1992.

Author: Ryan Strauss
Author: Sarah Hancock
"""
import numpy as np


class _Function:
    """Thin wrapper class around functions for integration with the GP algorithm."""

    def __init__(self, function, name, arity):
        if not callable(function):
            raise ValueError('function must be callable')

        if not (arity == 1 or arity == 2):
            raise ValueError('arity must be 1 or 2.')

        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        if len(args) != self.arity:
            raise ValueError('Arity of this function is {}, but {} were provided.'.format(self.arity, len(args)))
        return self.function(*args)


def _protected_division(x1, x2):
    """Closed division operation, as defined by [1]."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def _protected_exp(x):
    """Closed exponentiation operation."""
    return np.clip(np.exp(x), a_min=10e-10, a_max=10e10)


def _protected_log(x):
    """Closed logarithm operation, as defined by [1]."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x != 0, np.log(np.abs(x)), 0.)


def _protected_tan(x):
    """Closed tan operation."""
    return np.clip(np.tan(x), a_min=-10e10, a_max=10e10)


def _protected_sqrt(x):
    """Closed square root operation, as defined by [1]."""
    return np.sqrt(np.abs(x))


add = _Function(np.add, 'add', 2)
subtract = _Function(np.subtract, 'subtract', 2)
multiply = _Function(np.multiply, 'multiply', 2)
divide = _Function(_protected_division, 'divide', 2)
exp = _Function(_protected_exp, 'exp', 1)
log = _Function(_protected_log, 'log', 1)
sin = _Function(np.sin, 'sin', 1)
cos = _Function(np.cos, 'cos', 1)
tan = _Function(_protected_tan, 'tan', 1)
sqrt = _Function(_protected_sqrt, 'sqrt', 1)
