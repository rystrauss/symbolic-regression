import numpy as np

from ._program import _Program


class SymbolicRegressor:

    def __init__(self,
                 population_size,
                 function_set,
                 const_range,
                 int_consts,
                 tournament_size,
                 max_generations,
                 init_method,
                 fitness_function,
                 stopping_threshold,
                 standardized_fitness_max,
                 init_depth):
        # TODO: raise errors

        self.population_size = population_size
        self.function_set = function_set
        self.const_range = const_range
        self.int_consts = int_consts
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        self.init_method = init_method
        self.fitness_function = fitness_function
        self.stopping_threshold = stopping_threshold
        self.standardized_fitness_max = standardized_fitness_max
        self.init_depth = init_depth

        self.best_program = None

    def fit(self, X, y):
        # TODO: implement ramped initialization
        population = [_Program(self.function_set, self.init_depth, self.const_range, self.int_consts, X.shape[1],
                               self.init_method) for _ in range(self.population_size)]

        for generation in range(self.max_generations):
            raw_fitness = np.array([self.fitness_function(y, program.predict(X)) for program in population])

            if self.standardized_fitness_max:
                raw_fitness = self.standardized_fitness_max - raw_fitness
            standardized_fitness = raw_fitness

            adjusted_fitness = 1 / (1 + standardized_fitness)
            normalized_fitness = adjusted_fitness / adjusted_fitness.sum()

            normalized_fitness = list(enumerate(normalized_fitness))
            normalized_fitness.sort(key=lambda x: x[1])
