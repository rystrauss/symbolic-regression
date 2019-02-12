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
                 objective,
                 init_depth):

        self.population_size = population_size
        self.function_set = function_set
        self.const_range = const_range
        self.int_consts = int_consts
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        self.init_method = init_method
        self.fitness_function = fitness_function
        self.stopping_threshold = stopping_threshold
        self.objective = objective
        self.init_depth = init_depth

        self.best_program = None

        # TODO: deal with errors

    def fit(self, X, y):
        # TODO: implement ramped initialization
        population = [_Program(self.function_set, self.init_depth, self.const_range, self.int_consts, X.shape[1],
                               self.init_method) for _ in range(self.population_size)]

        for generation in range(self.max_generations):
            fitnesses = list(enumerate([self.fitness_function(y, program.predict(X)) for program in population]))
            fitnesses.sort(key=lambda x: x[1], reverse=self.objective == 'max')
