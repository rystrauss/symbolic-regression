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
                 operation_probabilities,
                 mutation_probabilities,
                 init_depth,
                 max_depth):
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
        self.operation_probabilities = operation_probabilities
        self.mutation_probabilities = mutation_probabilities
        self.init_depth = init_depth
        self.max_depth = max_depth

        self.best_program = None

    def fit(self, X, y):
        # TODO: implement ramped initialization

        history = []
        best_fitness = 0

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

            new_population = []

            best_index = np.argmax(adjusted_fitness)

            history.append({
                'best': raw_fitness[best_index],
                'average': np.mean(raw_fitness)
            })

            if adjusted_fitness[best_index] > best_fitness:
                self.best_program = population[best_index]

            # check termination criterion
            if adjusted_fitness[best_index] > 1 - self.stopping_threshold:
                break

            def tournament():
                contestents = np.random.choice(normalized_fitness, size=self.tournament_size, replace=False)
                contestents.sort(key=lambda x: x[1])
                return contestents[0][0]

            while len(new_population) < self.population_size:
                operation_probability = np.random.rand()

                # Clone
                if operation_probability < self.operation_probabilities[0]:
                    winner = population[tournament()]
                    new_population.append(winner.clone())

                # Crossover
                elif operation_probability < np.sum(self.operation_probabilities[:2]):
                    parent1 = population[tournament()]
                    parent2 = population[tournament()]

                    offspring1 = parent1.subtree_crossover(parent2)
                    offspring2 = parent2.subtree_crossover(parent1)

                    if offspring1.depth() > self.max_depth:
                        new_population.append(parent1.clone())

                    else:
                        new_population.append(offspring1)

                    if offspring2.depth() > self.max_depth:
                        new_population.append(parent2.clone())

                    else:
                        new_population.append(offspring2)

                # Mutations
                else:
                    winner = population[tournament()]

                    mutation_method = np.random.rand()

                    if mutation_method < self.mutation_probabilities[0]:
                        new_population.append(winner.hoist_mutation())

                    elif mutation_method < np.sum(self.mutation_probabilities[:2]):
                        new_population.append(winner.point_mutation())

                    else:
                        new_population.append(winner.subtree_mutation())

            population = new_population

        return history
