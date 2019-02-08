import numpy as np


class Function:

    def __init__(self, function, name, arity):
        if not callable(function):
            raise ValueError('function must be callable.')

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
    if np.abs(x2) < 0.001:
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


add = Function(np.add, 'add', 2)
subtract = Function(np.subtract, 'subtract', 2)
multiply = Function(np.multiply, 'multiply', 2)
divide = Function(_protected_division, 'divide', 2)
exp = Function(_protected_exp, 'exp', 1)
log = Function(_protected_log, 'log', 1)
sin = Function(np.sin, 'sin', 1)
cos = Function(np.cos, 'cos', 1)
tan = Function(_protected_tan, 'tan', 1)
# TODO: Add sqrt
