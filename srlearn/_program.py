"""The underlying data structure used in the genetic programming algorithm.

A program represents one individual in the population and uses prefix notation to
represent an expression tree.

Parts of this implementation are adapted from gplearn:
https://github.com/trevorstephens/gplearn

Author: Ryan Strauss
Author: Sarah Hancock
"""

from copy import deepcopy

import numpy as np

from .functions import _Function


class _Program:
    """A prefix notation representation of a program.

    This is the data structure that gets operated on by the genetic algorithm.
    """

    def __init__(self,
                 function_set,
                 init_depth,
                 const_range,
                 int_consts,
                 num_features,
                 init_method,
                 program=None):
        """Constructor.

        Args:
            function_set (list): A list of valid functions to use in the program.
            init_depth (int): The maximum allowed initialization depth for the tree when the initialization
            method is 'grow'. When the initialization method is 'full', this will be the height of the tree.
            const_range (tuple of two ints): The range of constants to include in the formulas.
            int_consts (bool): If true, constants will only be integers in const_range.
            num_features (int): The number of features.
            init_method (str):
                - 'grow': Nodes are chosen at random from both functions and terminals, allowing
                          for smaller trees than `init_depth` allows. Tends to grow asymmetrical trees.
                - 'full': Functions are chosen until the `max_depth` is reached, and then terminals are selected.
                          Tends to grow 'bushy' trees.
            program (tuple, optional): The prefix notation representation of the program. If None, a new naive
            random tree will be grown.
        """
        if init_depth < 1:
            raise ValueError('max_depth must be at least 1.')
        if not isinstance(const_range, tuple) or len(const_range) != 2:
            raise ValueError('terminal_range must be a 2-tuple.')
        if num_features < 1:
            raise ValueError('num_features must be at least 1.')
        if init_method not in ['grow', 'full']:
            raise ValueError('"{}" is not a valid init_method.'.format(init_method))
        for element in function_set:
            if not isinstance(element, _Function):
                raise ValueError('function_set can only contain elements of type `Function`.')

        self.function_set = function_set
        self.init_depth = init_depth
        self.const_range = const_range
        self.int_consts = int_consts
        self.num_features = num_features
        self.init_method = init_method
        self.program = program or self.generate_random_program()

        if not isinstance(self.program, list):
            raise ValueError('program must be a list.')

        arities = {}
        for function in self.function_set:
            if function.arity not in arities:
                arities[function.arity] = []
            arities[function.arity].append(function)

        self.arities = arities

    def __str__(self):
        """Returns a prefix notation string representation of the program."""
        output = ''
        terminals = [0]
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
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

    def __len__(self):
        """Defines the length of a program to be the number of nodes in the program."""
        return len(self.program)

    def depth(self):
        """Calculates the depth of the program."""
        return _depth(self.program)

    def generate_random_program(self, max_depth=None):
        """Generates a random program.

        Builds a random tree using either the 'full' or 'grow' method, as specified in
        `self.init_method`. These two techniques are described in [1] and this implementation
        is adapted from the presentation of the algorithm in [2].

        [1] J. R. Koza, Genetic programming: on the programming of computers by means of natural selection. 1992.
        [2] W. B. Langdon, R. Poli, N. F. McPhee, and J. R. Koza, “Genetic programming: An introduction and tutorial,
        with a survey of techniques and applications,” Stud. Comput. Intell., vol. 115, pp. 927–1028, 2008.

        Args:
            max_depth (int, optional): The maximum depth of the generated program. Defaults to None,
            in which case the max depth of this program object is used.

        Returns:
            The explicit prefix notation representation of the generated tree as a list.
        """
        # Start program with a function
        function = np.random.choice(self.function_set)
        program = [function]
        terminal_stack = [function.arity]

        max_depth = max_depth or self.init_depth

        while terminal_stack:
            depth = len(terminal_stack)
            # We consider `rand` to be a single terminal, so the size of our terminal set
            # is the number of features plus one
            terminal_prob = (self.num_features + 1) / (len(self.function_set) + self.num_features + 1)
            if depth == max_depth or (self.init_method == 'grow' and np.random.rand() <= terminal_prob):
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
        """Makes predictions for the provided examples using the current program.

        Args:
            X (ndarray): A numpy array of the shape (num_examples, num_features).

        Returns:
            The predicted values as a numpy array of the shape (num_examples,).
        """
        if X.ndim != 2 or X.shape[1] != self.num_features:
            raise ValueError(
                'X should have shape (num_examples, {}), but got shape {}.'.format(self.num_features, X.shape))

        evaluation_stack = []
        for node in self.program:
            if isinstance(node, _Function):
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

    def clone(self):
        """Clones the program.

        Returns:
            A deep copy of the program.
        """
        return deepcopy(self)

    def subtree_crossover(self, donor):
        """Performs subtree crossover on the program.

        Given two parents, subtree crossover randomly selects a crossover point in each parent tree.
        Then, it creates the offspring by replacing the sub-tree rooted at the crossover point in a copy
        of the first parent with a copy of the sub-tree rooted at the crossover point in the second parent.

        This implementation restricts the offspring from being more than 15% deeper than its parent, as proposed by [1].

        [1] K. E. Kinnear, “Evolving a sort: lessons in genetic programming,” in IEEE International Conference on
        Neural Networks, 1993, pp. 881–888.

        Args:
            donor (_Program): The donor program.

        Returns:
            The offspring that resulted from the crossover.
        """
        if not isinstance(donor, _Program):
            raise ValueError('donor must be a _Program.')

        # Produce offspring
        offspring = self.clone()
        offspring_depth = offspring.depth()

        # Get a subtree of the offspring to replace
        start, end = _get_random_subtree(offspring.program)

        # Get a subtree to donate
        offset = offspring_depth - _depth(offspring.program[start:end])
        donor_start, donor_end = _get_random_subtree(donor.program)
        donor_program = donor.program[donor_start:donor_end]
        # If depth is too big, try again
        while offset + _depth(donor_program) > offspring_depth * 1.15:
            donor_start, donor_end = _get_random_subtree(donor_program)
            donor_program = donor_program[donor_start:donor_end]

        # Insert genetic material from the donor into the offspring
        offspring.program = offspring.program[:start] + donor_program + offspring.program[end:]

        return offspring

    def subtree_mutation(self):
        """Performs a subtree mutation on the program.

        Subtree mutation is the most common form of GP mutation. This method randomly selects a
        mutation point in a tree and substitutes the sub-tree rooted there with a randomly
        generated sub-tree. Subtree mutation is sometimes implemented as crossover between a program
        and a newly generated random program; this operation is also known as ‘headless chicken’ crossover.

        W. B. Langdon, R. Poli, N. F. McPhee, and J. R. Koza, “Genetic programming: An introduction and
        tutorial, with a survey of techniques and applications,” Stud. Comput. Intell., vol. 115, pp. 927–1028, 2008.

        Returns:
            The mutated program.
        """
        # Build a new naive program
        chicken = self.clone()
        chicken.program = self.generate_random_program()
        # Do subtree mutation via the headless chicken method
        return self.subtree_crossover(chicken)

    def point_mutation(self, point_probability):
        """Performs point mutation on the program.

        In point mutation a random node is selected and the primitive stored there is replaced with a different
        random primitive of the same arity taken from the primitive set. If no other primitives with that arity
        exist, nothing happens to that node (but other nodes may still be mutated). Note that, when subtree mutation
        is applied, this involves the modification of exactly one subtree. Point mutation, on the other hand, is
        typically applied with a given mutation rate on a per-node basis, allowing multiple nodes to be mutated
        independently.

        W. B. Langdon, R. Poli, N. F. McPhee, and J. R. Koza, “Genetic programming: An introduction and
        tutorial, with a survey of techniques and applications,” Stud. Comput. Intell., vol. 115, pp. 927–1028, 2008.

        Args:
            point_probability (float): The probability of a single node being mutated.

        Returns:
            The mutated program.
        """
        mutated = self.clone()
        # Determine which nodes to mutate
        indices = np.where(
            [True if np.random.rand() < point_probability else False for _ in range(len(mutated.program))])

        for i in indices:
            node = mutated.program[i]
            if isinstance(node, _Function):
                # If node is a function, replace it with a random function of equal arity
                # Note that there is a chance the same function is randomly selected as a replacement
                mutated.program[i] = np.random.choice(mutated.arities[node.arity])
            else:
                # We need to select either a variable or constant
                terminal = np.random.randint(mutated.num_features + 1)
                if terminal == mutated.num_features:
                    terminal = np.random.uniform(*self.const_range)
                mutated.program[i] = terminal

        return mutated

    def hoist_mutation(self):
        """Performs Hoist mutation on the program.

        In Hoist mutation the new subtree is selected from the subtree being removed from the parent,
        guaranteeing that the new program will be smaller than its parent. Hoist mutation is a method
        for controlling bloat.

        [1] W. B. Langdon, R. Poli, N. F. McPhee, and J. R. Koza, “Genetic programming: An introduction and
        tutorial, with a survey of techniques and applications,” Stud. Comput. Intell., vol. 115, pp. 927–1028, 2008.
        [2] K. E. Kinnear, “Generality and difficulty in genetic programming: Evolving a sort,” Proc. 5th Int. Conf.
        Genet. Algorithms (ICGA ’93), pp. 287–294, 1993.

        Returns:
            The mutated program.
        """
        mutated = self.clone()

        start, end = _get_random_subtree(mutated.program)
        sub_start, sub_end = _get_random_subtree(mutated.program[start:end])
        mutated.program = mutated.program[:start] + mutated.program[sub_start:sub_end] + mutated.program[:end]

        return mutated


def _depth(program):
    """Calculates the depth of a program.

    Args:
        program (list): The program for which to calculate the depth.

    Returns:
        The depth of the tree.
    """
    terminals = [0]
    depth = 1
    for node in program:
        if isinstance(node, _Function):
            terminals.append(node.arity)
            depth = max(len(terminals), depth)
        else:
            terminals[-1] -= 1
            while terminals[-1] == 0:
                terminals.pop()
                terminals[-1] -= 1
    return depth - 1


def _get_random_subtree(program):
    """Get a random subtree from the program.

    This method uses a technique suggested by Koza, where there is a 90%
    chance of a function getting selected and a 10% change of a terminal getting
    selected. On the other hand, if uniform selection of crossover points was used,
    crossover operations would frequently exchange only very small amounts of genetic
    material (that is, small subtrees); many crossovers may in fact reduce to simply
    swapping two leaves.

    W. B. Langdon, R. Poli, N. F. McPhee, and J. R. Koza, “Genetic programming: An introduction and
    tutorial, with a survey of techniques and applications,” Stud. Comput. Intell., vol. 115, pp. 927–1028, 2008.

    Args:
        program (list): The explicit list representation of the program to get a subtree
        from.

    Returns:
        The tuple (start, end) representing the indices that mark the subtree. The endpoint is not inclusive.
    """
    probs = np.array([0.9 if isinstance(node, _Function) else 0.1 for node in program])
    probs = np.cumsum(probs / probs.sum())
    start = np.searchsorted(probs, np.random.uniform())

    # Keep track of the number of arguments we need to encapsulate
    stack = 1
    end = start
    # Check if we are encapsulated everything we need to
    while stack > end - start:
        node = program[end]
        if isinstance(node, _Function):
            # If we are at a function, we need to encapsulate its children
            stack += node.arity
        # Push back the endpoint
        end += 1

    return start, end
