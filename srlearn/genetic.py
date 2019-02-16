"""Contains an implementation of genetic programming for symbolic regression.

Author: Ryan Strauss
Author: Sarah Hancock
"""
import numpy as np

from ._program import _Program
from .fitness import mean_squared_error, _Fitness
from .functions import add, subtract, multiply, divide, _Function


class SymbolicRegressor:
    """A class for performing symbolic regression."""

    def __init__(self,
                 population_size=500,
                 function_set=(add, subtract, multiply, divide),
                 const_range=(-5, 5),
                 int_consts=True,
                 tournament_size=5,
                 init_method='ramped',
                 fitness_function=mean_squared_error,
                 crossover_probability=0.9,
                 mutation_probability=0.05,
                 clone_probability=0.05,
                 mutation_type_probabilities=(0.5, 0.25, 0.25),
                 point_replace_probability=0.05,
                 init_depth=6,
                 max_depth=17,
                 parsimony_coefficient=0.001,
                 random_state=None):
        """Constructor.

        Args:
            population_size (int): The size of the population.
            function_set (tuple): A tuple containing the functions that should be used to construct programs.
            const_range (tuple): A tuple of the form (min, max), specifying the range of allowable constants.
            int_consts (bool): If true, only integer constants will be used.
            tournament_size (int): The number of contestant programs in the tournaments.
            init_method (str): One of 'full', 'grow', or 'ramped', specifying how the original programs should
            be initialized.
            fitness_function (_Fitness): A valid fitness function from `srlearn.fitness`.
            crossover_probability (float): The probability of a crossover occurring on a given iteration.
            mutation_probability (float): The probability of a mutation occurring on a given iteration.
            clone_probability (float): The probability of reproduction occurring on a given iteration.
            mutation_type_probabilities (tuple): A tuple specifying the probabilities of each of the three
            mutations occurring. The elements in the tuple correspond to (hoist, point, subtree) mutation. These values
            must add to 1.
            point_replace_probability (float): When point mutation occurs, this is the probability of any given point
            being mutated.
            init_depth (int): The depth at which trees should be initialized.
            max_depth (int): The maximum depth allowed in the population.
            parsimony_coefficient (float): Coefficient to determine how much parsimony pressure (i.e. regularization)
            is applied to the fitness during tournaments. A greater value means more pressure.
        """
        if not isinstance(function_set, tuple):
            raise ValueError('function set must be a tuple.')
        for f in function_set:
            if not isinstance(f, _Function):
                raise ValueError('{} is not a valid function.'.format(f))
        if init_method not in ['full', 'grow', 'ramped']:
            raise ValueError('invalid init method.')
        if not isinstance(fitness_function, _Fitness):
            raise ValueError('provided fitness function is not valid.')
        if sum((crossover_probability, mutation_probability, clone_probability)) != 1:
            raise ValueError('sum of crossover, mutation, and clone probabilities must equal 1.')
        if sum(mutation_type_probabilities) != 1:
            raise ValueError('sum of mutation type probabilities must equal 1.')

        self.population_size = population_size
        self.function_set = function_set
        self.const_range = const_range
        self.int_consts = int_consts
        self.tournament_size = tournament_size
        self.init_method = init_method
        self.fitness_function = fitness_function
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.mutation_type_probabilities = mutation_type_probabilities
        self.point_replace_probability = point_replace_probability
        self.init_depth = init_depth
        self.max_depth = max_depth
        self.parsimony_coefficient = parsimony_coefficient
        self.random_state = random_state or np.random.RandomState()

        self.best_program = None

    def fit(self, X, y, max_generations=50, stopping_threshold=0.01):
        """Fits the model to the provided data by running the genetic programming algorithm.

        Args:
            X (ndarray): A numpy array of the form (num_samples, num_features).
            y (ndarray): A numpy array of the form (num_samples,) containing the true y values.
            max_generations (int): The maximum number of allowed generations.
            stopping_threshold (float): A stopping criterion based on the best adjusted fitness score in the population.
            If the best adjusted fitness is within this threshold from 1, the program terminates.

        Returns:
            A list containing a history of metrics about the training process.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError('X must be a numpy array.')
        if not isinstance(y, np.ndarray):
            raise ValueError('y must be a numpy array.')
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        if X.ndim != 2:
            raise ValueError('X should be two dimensional')

        history = {
            'raw_fitness': {
                'best': [],
                'average': []
            },
            'standardized_fitness': {
                'best': [],
                'average': []
            },
            'adjusted_fitness': {
                'best': [],
                'average': []
            },
            'diversity': [],
            'best_raw_fitness': None
        }

        # Build the initial population
        if self.init_method == 'ramped':
            population = []
            for depth in range(2, self.init_depth + 1):
                population.extend(
                    [_Program(self.function_set, depth, self.const_range, self.int_consts, X.shape[1], self.init_method,
                              self.random_state) for _ in range(self.population_size // self.init_depth)])
            if len(population) < self.population_size:
                population.extend(
                    [_Program(self.function_set, self.init_depth, self.const_range, self.int_consts,
                              X.shape[1], self.init_method, self.random_state) for _ in
                     range(self.population_size - len(population))])
        else:
            population = [_Program(self.function_set, self.init_depth, self.const_range, self.int_consts, X.shape[1],
                                   self.init_method, self.random_state) for _ in range(self.population_size)]

        assert len(population) == self.population_size

        best_adjusted_fitness = 0

        for generation in range(max_generations):
            for program in population:
                assert program.is_valid()

            # Calculate the raw fitness scores
            raw_fitness = np.array([self.fitness_function(y, program.predict(X)) for program in population])

            # Standardize the fitness scores
            standardized_fitness = raw_fitness
            if self.fitness_function.maximize:
                standardized_fitness = self.fitness_function.max_value - raw_fitness

            # Calculate the adjusted fitness scores
            adjusted_fitness = 1 / (1 + standardized_fitness)

            # Find the index of the program with the best fitness
            best_index = np.argmax(adjusted_fitness)
            assert isinstance(best_index, np.int64)

            # Save fitness metrics to history
            history['raw_fitness']['best'].append(raw_fitness[best_index])
            history['raw_fitness']['average'].append(raw_fitness.mean())
            history['standardized_fitness']['best'].append(standardized_fitness[best_index])
            history['standardized_fitness']['average'].append(standardized_fitness.mean())
            history['adjusted_fitness']['best'].append(adjusted_fitness[best_index])
            history['adjusted_fitness']['average'].append(adjusted_fitness.mean())
            history['diversity'].append(_diversity(population))

            # Update the overall best program if necessary
            if adjusted_fitness[best_index] > best_adjusted_fitness:
                self.best_program = population[best_index]
                history['best_raw_fitness'] = raw_fitness[best_index]

            # Check termination criterion
            if adjusted_fitness[best_index] > 1 - stopping_threshold:
                break

            def tournament():
                """Holds a tournament between randomly selected programs.

                Returns:
                    The index of the winning program.
                """
                contestant_indices = self.random_state.choice(self.population_size, size=self.tournament_size,
                                                              replace=False)
                # Apply parsimony pressure
                regularized_fitness = np.array([f - self.parsimony_coefficient * len(population[i]) for i, f in
                                                enumerate(adjusted_fitness)])
                contestants = regularized_fitness[contestant_indices]
                return contestant_indices[np.argmax(contestants)]

            # Initialize the new population
            new_population = []
            # Loop until the new population has been filled
            while len(new_population) < self.population_size:
                # Randomly determine which operation is going to happen
                operation_probability = self.random_state.rand()

                # Mutation
                if operation_probability < self.mutation_probability:
                    winner = population[tournament()]
                    mutation_method = self.random_state.rand()
                    # Hoist mutation
                    if mutation_method < self.mutation_type_probabilities[0]:
                        new_population.append(winner.hoist_mutation())
                    # Point mutation
                    elif mutation_method < np.sum(self.mutation_type_probabilities[:2]):
                        new_population.append(winner.point_mutation(self.point_replace_probability))
                    # Subtree mutation
                    else:
                        mutated = winner.subtree_mutation()
                        # Ensure we have not exceeded the maximum depth
                        if mutated.depth() > self.max_depth:
                            new_population.append(winner.clone())
                        else:
                            new_population.append(mutated)
                # Crossover
                elif operation_probability < self.mutation_probability + self.crossover_probability:
                    parent1 = population[tournament()]
                    parent2 = population[tournament()]

                    offspring1 = parent1.subtree_crossover(parent2)
                    offspring2 = parent2.subtree_crossover(parent1)

                    # Ensure we have not exceeded the maximum depth
                    if offspring1.depth() > self.max_depth:
                        new_population.append(parent1.clone())
                    else:
                        new_population.append(offspring1)
                    if offspring2.depth() > self.max_depth:
                        new_population.append(parent2.clone())
                    else:
                        new_population.append(offspring2)
                # Clone
                else:
                    winner = population[tournament()]
                    new_population.append(winner.clone())

            population = new_population

        return history

    def predict(self, X):
        """Make predictions using the best program.

        Args:
            X (ndarray): A numpy array of the examples to predict on.

        Returns:
            A numpy array with the predicted values.
        """
        if self.best_program is None:
            raise RuntimeError('regressor has not yet been fitted.')
        if not isinstance(X, np.ndarray):
            raise ValueError('X must be a numpy array.')
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        if X.ndim != 2:
            raise ValueError('X should be two dimensional')

        return self.best_program.predict(X)

    def score(self, X, y):
        if self.best_program is None:
            raise RuntimeError('regressor has not yet been fitted.')
        if not isinstance(X, np.ndarray):
            raise ValueError('X must be a numpy array.')
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        if X.ndim != 2:
            raise ValueError('X should be two dimensional')
        if not isinstance(y, np.ndarray):
            raise ValueError('y must be a numpy array.')

        return self.fitness_function(y, self.predict(X))


def _diversity(population):
    """Calculates the diversity of a population.

    Args:
        population (list): The population to evaluate.

    Returns:
        The percentage of individuals for which no exact duplicate exists elsewhere in the population.
    """
    population = [str(p) for p in population]
    frequencies = [population.count(p) for p in set(population)]
    return frequencies.count(1) / len(population)
