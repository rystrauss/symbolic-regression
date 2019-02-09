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
        if max_depth < 1:
            raise ValueError('max_depth must be at least 1.')
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
        self.program = program or self.generate_random_program()

        if not isinstance(self.program, list):
            raise ValueError('program must be a list.')

    def __str__(self):
        output = ''
        terminals = [0]
        for i, node in enumerate(self.program):
            if isinstance(node, Function):
                output += node.name + '('
                terminals.append(node.arity)
            else:
                if isinstance(node, np.int):
                    output += 'X{}'.format(node)
                else:
                    output += '{0:.3f}'.format(node)
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '

        return output

    def generate_random_program(self):
        """Generates a random program.

        Builds a random tree using either the 'full' or 'grow' method, as specified in
        `self.init_method`. These two techniques are described in [1] and this implementation
        is adapted from the presentation of the algorithm in [2].

        [1] J. R. Koza, Genetic programming: on the programming of computers by means of natural selection. 1992.
        [2] W. B. Langdon, R. Poli, N. F. McPhee, and J. R. Koza, “Genetic programming: An introduction and tutorial,
        with a survey of techniques and applications,” Stud. Comput. Intell., vol. 115, pp. 927–1028, 2008.

        Returns:
            The explicit prefix notation representation of the generated tree as a list.
        """
        # Start program with a function
        function = np.random.choice(self.function_set)
        program = [function]
        terminal_stack = [function.arity]

        while terminal_stack:
            depth = len(terminal_stack)
            terminal_prob = (self.num_features + 1) / (len(self.function_set) + self.num_features + 1)
            if depth == self.max_depth or (self.init_method == 'grow' and np.random.rand() <= terminal_prob):
                # We need a terminal
                terminal = np.random.randint(self.num_features + 1)
                # Potentially select a constant as the terminal
                if terminal == self.num_features:
                    terminal = np.random.uniform(*self.const_range)
                    if self.int_consts:
                        terminal = np.round(terminal)
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1
            else:
                # We need a function
                function = np.random.choice(self.function_set)
                program.append(function)
                terminal_stack.append(function.arity)

    def predict(self, X):
        if X.ndim != 2 or X.shape[1] != self.num_features:
            raise ValueError(
                'X should have shape (num_examples, {}), but got shape {}.'.format(self.num_features, X.shape))

        evaluation_stack = []
        for node in self.program:
            if isinstance(node, Function):
                # If node is a function, push a new list onto the evaluation stack
                evaluation_stack.append([node])
            else:
                # If node is a terminal, append it to its corresponding function's list
                evaluation_stack[-1].append(node)
            # Check for functions that are ready to be evaluated
            while len(evaluation_stack[-1]) == evaluation_stack[-1][0].arity + 1:
                function = evaluation_stack[-1][0]
                terminals = []
                for t in evaluation_stack[-1][1:]:
                    if isinstance(t, np.float):
                        # Terminal is a constant
                        terminals.append(np.full(X.shape[0], t, dtype=np.float))
                    elif isinstance(t, np.int):
                        # Terminal is a variable:
                        terminals.append(X[:, t])
                    else:
                        # Terminal is a numpy array containing a previously computed result
                        terminals.append(t)
                # Evaluate the function
                result = function(*terminals)
                # Check to see if we have returned to the root
                if len(evaluation_stack) != 1:
                    # If not, pop the now completed function and add the result
                    # as an argument for the next function
                    evaluation_stack.pop()
                    evaluation_stack[-1].append(result)
                else:
                    # In this case, we are done and can return the result
                    return result

    def get_random_subtree(self, depth=None):
        """Recursivley traversed the tree to get a random subtree.

        This function stops at the parent of the selected subtree, so the subtree can be swapped out
        and references work as needed.

        Args:
            depth (int, optional): The maximal depth at which to pick a subtree. If None, the depth
            is randomly selected.

        Returns:
            A tuple where the first element is the parent of the selected subtree,
            the second element is the index of the selected subtree in its parent, and the
            third element is the depth at which the subtree is located.
        """
        raise NotImplementedError

    def subtree_mutation(self):
        """Performs a subtree mutation on the program.

        Subtree mutation is the most common form of GP mutation. This method randomly selects a
        mutation point in a tree and substitutes the sub-tree rooted there with a randomly
        generated sub-tree. Subtree mutation is sometimes implemented as crossover between a program
        and a newly generated random program; this operation is also known as ‘headless chicken’ crossover.

        W. B. Langdon, R. Poli, N. F. McPhee, and J. R. Koza, “Genetic programming: An introduction and
        tutorial, with a survey of techniques and applications,” Stud. Comput. Intell., vol. 115, pp. 927–1028, 2008.

        Returns:
            None
        """
        raise NotImplementedError
