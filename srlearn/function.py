import numpy as np


class Function:

    def __init__(self, function, name, arity):
        if not isinstance(function, np.ufunc):
            raise ValueError('function must be a numpy ufunc.')

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
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def _protected_exp(x):
    return np.clip(np.exp(x), a_min=10e-10, a_max=10e10)


def _protected_log(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x != 0, np.log(np.abs(x)), 0.)


def _protected_tan(x):
    return np.clip(np.tan(x), a_min=-10e10, a_max=10e10)


def _protected_sqrt(x):
    return np.sqrt(np.abs(x))


add = Function(np.add, 'add', 2)
subtract = Function(np.subtract, 'subtract', 2)
multiply = Function(np.multiply, 'multiply', 2)
divide = Function(_protected_division, 'divide', 2)
exp = Function(_protected_exp, 'exp', 1)
log = Function(_protected_log, 'log', 1)
sin = Function(np.sin, 'sin', 1)
cos = Function(np.cos, 'cos', 1)
tan = Function(_protected_tan, 'tan', 1)
sqrt = Function(_protected_sqrt, 'sqrt', 1)
