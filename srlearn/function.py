import numpy as np


class _Function:

    def __init__(self, function, arity):
        if not isinstance(function, callable):
            raise ValueError('Function must be callable.')

        self.function = function
        self.arity = arity

    def __call__(self, *args, **kwargs):
        if len(args) != self.arity:
            raise ValueError('Arity of this function is {}, but {} were provided.'.format(self.arity, len(args)))
        return self.function(*args)


def _protected_division(x1, x2):
    if x2 == 0:
        return 1.
    return np.divide(x1, x2)


def _protected_exp(x):
    y = np.exp(x)
    if y > 10e10:
        return 10e10
    if y < 10e-10:
        return 10e-10
    return y


def _protected_log(x):
    if x == 0:
        return 0
    return np.log(np.abs(x))  # TODO: Come back to this


def _protected_tan(x):
    y = np.exp(x)
    if y > 10e10:
        return 10e10
    if y < -10e10:
        return -10e10
    return y


add = _Function(np.add, 2)
subtract = _Function(np.subtract, 2)
multiply = _Function(np.multiply, 2)
divide = _Function(_protected_division, 2)
exp = _Function(_protected_exp, 1)
log = _Function(_protected_log, 1)
sin = _Function(np.sin, 1)
cos = _Function(np.cos, 1)
tan = _Function(_protected_tan, 1)
