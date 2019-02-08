"""The underlying data structure used in the genetic programming algorithm.

A program represents one individual in the population and uses prefix notation to
represent an expression tree.

Author: Ryan Strauss
Author: Sarah Hancock
"""

import numpy as np

from .function import Function


class _Program:
    """A prefix notation representation of a program.

    This is the data structure that gets operated on by the genetic algorithm.
    """

    def __init__(self,
                 function_set,
                 max_depth,
                 const_range,
                 int_consts,
                 num_features,
                 init_method,
                 program=None):
        """Constructor.

        Args:
            function_set (list): A list of valid functions to use in the program.
            max_depth (int): The maximum allowed depth for the tree when the initialization
            method is 'grow'. When the initialization method is 'full', this will be the height of the tree.
            const_range (tuple of two ints): The range of constants to include in the formulas.
            int_consts (bool): If true, constants will only be integers in const_range.
            num_features (int): The number of features.
            init_method (str):
                - 'grow': Nodes are chosen at random from both functions and terminals, allowing
                          for smaller trees than `init_depth` allows. Tends to grow asymmetrical trees.
                - 'full': Functions are chosen until the `init_depth` is reached, and then terminals are selected.
                          Tends to grow 'bushy' trees.
            program (tuple, optional): The prefix notation representation of the program. If None, a new naive
            random tree will be grown.
        """
        if max_depth < 0:
            raise ValueError('max_depth must be at least zero.')
        if not isinstance(const_range, tuple) or len(const_range) != 2:
            raise ValueError('terminal_range must be a 2-tuple.')
        if num_features < 1:
            raise ValueError('num_features must be at least 1.')
        if init_method not in ['grow', 'full']:
            raise ValueError('"{}" is not a valid init_method.'.format(init_method))
        for element in function_set:
            if not isinstance(element, Function):
                raise ValueError('function_set can only contain elements of type `Function`.')

        self.function_set = function_set
        self.max_depth = max_depth
        self.const_range = const_range
        self.int_consts = int_consts
        self.num_features = num_features
        self.init_method = init_method
        self.program = program or self.generate_random_program(self.max_depth)

    def generate_random_program(self, depth):
        """Recursivley generates a random program.

        Builds a random tree using either the 'full' or 'grow' method, as specified in
        `self.init_method`. These two techniques are described in [1] and this implementation
        follows the presentation of the algorithm in [2].

        [1] J. R. Koza, Genetic programming: on the programming of computers by means of natural selection. 1992.
        [2] W. B. Langdon, R. Poli, N. F. McPhee, and J. R. Koza, “Genetic programming: An introduction and tutorial,
        with a survey of techniques and applications,” Stud. Comput. Intell., vol. 115, pp. 927–1028, 2008.

        Args:
            depth (int): The maximum depth of the tree to be created.

        Returns:
            The prefix notation representation of the generated tree as a tuple.
        """
        if depth > self.max_depth:
            raise ValueError('depth cannot be larger than the maximum depth.')

        term_size = self.num_features + 1
        func_size = len(self.function_set)
        # Determine if a function or terminal should be added
        if depth == 0 or (
                self.init_method == 'grow' and np.random.rand() < term_size / (term_size + func_size)):
            # We need to select a terminal
            terminal = np.random.randint(term_size)
            # Potentially select a constant as the terminal
            if terminal == self.num_features:
                terminal = np.random.uniform(*self.const_range)
                if self.int_consts:
                    terminal = np.round(terminal)
            program = terminal
        else:
            # We need to select a function
            function = np.random.choice(self.function_set)
            # Recursivley generate function arguments
            arguments = [self.generate_random_program(depth - 1) for _ in range(function.arity)]
            program = [function, *arguments]

        return program

    def subtree_mutation(self):
        depth = np.random.randint(self.max_depth // 2, self.max_depth)
        cur = self.program
        for _ in range(depth):
            # See if we've reached a terminal
            if not isinstance(property, tuple):
                new_subtree = self.generate_random_program(self.max_depth - depth)
