"""Script used for running GP experiments in parallel.

Author: Ryan Strauss
"""
import json
import multiprocessing
import os
import pickle

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from srlearn import SymbolicRegressor
from srlearn.fitness import *
from srlearn.functions import *

FITNESS = [mean_squared_error, mean_absolute_error, r_squared]


def perform_run(run_id, population_size, fitness, function_set, parsimony_coefficient, const_range, tournament_size,
                int_consts, crossover_probability, mutation_probability, clone_probability, init_depth, X_train,
                y_train, stopping_threshold, max_generations, save_dir, X_test, y_test):
    """Performs a single run of GP. This is the worker function that gets run by parallel processes."""
    print('Starting run {}...'.format(run_id))
    model = SymbolicRegressor(population_size=population_size,
                              fitness_function=FITNESS[fitness],
                              function_set=function_set,
                              parsimony_coefficient=parsimony_coefficient,
                              const_range=const_range,
                              tournament_size=tournament_size,
                              int_consts=int_consts,
                              crossover_probability=crossover_probability,
                              mutation_probability=mutation_probability,
                              clone_probability=clone_probability,
                              init_depth=init_depth)

    # Fit the model to the data
    history = model.fit(X_train, y_train, stopping_threshold=stopping_threshold, max_generations=max_generations)

    # Save model and history
    if not os.path.exists(os.path.join(save_dir, 'run_{}'.format(run_id))):
        os.makedirs(os.path.join(save_dir, 'run_{}'.format(run_id)))
    with open(os.path.join(save_dir, 'run_{}'.format(run_id), 'history.json'), 'w') as fp:
        json.dump(history, fp)
    with open(os.path.join(save_dir, 'run_{}'.format(run_id), 'model.p'), 'wb') as fp:
        pickle.dump(model, fp)

    # Return the run ID and the model's performance on the test set
    return run_id, model.score(X_test, y_test)


@click.command()
@click.argument('data_path', type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=1)
@click.argument('save_dir', type=click.Path(exists=False, file_okay=False, dir_okay=True), nargs=1)
@click.option('--population_size', type=click.INT, nargs=1, default=500,
              help='The number of individuals in the population.')
@click.option('--parsimony_coefficient', type=click.FLOAT, nargs=1, default=0.0001,
              help='Constant that controls the amount of parsimony pressure. '
                   'A higher value indicates greater pressure.')
@click.option('--stopping_threshold', type=click.FLOAT, nargs=1, default=0.001,
              help='The threshold at which a run will terminate. The value refers to the adjusted '
                   'fitneess\' acceptable distance from 1.')
@click.option('--max_generations', type=click.INT, nargs=1, default=50,
              help='The maximum number of generations allowed in a single run.')
@click.option('--const_range', type=click.INT, nargs=2, default=(-5, 5),
              help='The range in which constants will be generated.')
@click.option('--tournament_size', type=click.INT, nargs=1, default=5,
              help='The number of individuals that compete in the tournament selection.')
@click.option('--int_consts', type=click.BOOL, nargs=1, default=False, help='If true, constants will be integers.')
@click.option('--crossover_probability', type=click.FLOAT, nargs=1, default=0.9,
              help='The probability of crossover occurring.')
@click.option('--mutation_probability', type=click.FLOAT, nargs=1, default=0.02,
              help='The probability of mutation occurring.')
@click.option('--clone_probability', type=click.FLOAT, nargs=1, default=0.08,
              help='The probability of reproduction occurring.')
@click.option('--init_depth', type=click.INT, nargs=1, default=6,
              help='The initialization depth for the original population.')
@click.option('--include_analytic', type=click.BOOL, nargs=1, default=False,
              help='If true, a few analytic functions will be included in the function set.')
@click.option('--num_runs', type=click.INT, nargs=1, default=1, help='The number of GP runs to be executed.')
@click.option('--fitness', type=click.INT, nargs=1, default=0,
              help='Specifies the fitness function to use. 0 gives mean squared error, 1 gives '
                   'mean absolute error, and 2 gives R^2.')
@click.option('--workers', type=click.INT, nargs=1, default=1,
              help='The number of cores to use for multiprocessing. If -1, all cores will be used.')
def main(data_path, save_dir, population_size, parsimony_coefficient, stopping_threshold, max_generations, const_range,
         tournament_size, int_consts, crossover_probability, mutation_probability, clone_probability, init_depth,
         include_analytic, num_runs, fitness, workers):
    """Performs multiple runs of symbolic regression in parallel."""
    # Load in the data
    data = pd.read_csv(data_path)
    X = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1])

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

    function_set = (add, subtract, multiply, divide, exp, cos, sin) if include_analytic else (
        add, subtract, multiply, divide)

    if workers == -1:
        workers = None

    # Set up multiprocessing pool
    with multiprocessing.Pool(processes=workers) as p:
        # Execute GP runs in parallel
        test_scores = [p.apply_async(
            perform_run,
            args=(run_id, population_size, fitness, function_set, parsimony_coefficient, const_range, tournament_size,
                  int_consts, crossover_probability, mutation_probability, clone_probability, init_depth, X_train,
                  y_train, stopping_threshold, max_generations, save_dir, X_test, y_test))
            for run_id in range(num_runs)]
        for result in test_scores:
            result.wait()

    test_scores = [x.get() for x in test_scores]
    test_scores.sort(key=lambda x: x[0])
    test_scores = [x[1] for x in test_scores]

    # Save summary of all runs to file
    with open(os.path.join(save_dir, 'results.txt'), 'w') as fp:
        fp.write('*****PARAMETERS*****\n')
        fp.write('Data Path: {}\n'.format(data_path))
        fp.write('Save Path: {}\n'.format(save_dir))
        fp.write('Population Size: {}\n'.format(population_size))
        fp.write('Parsimony Coefficient: {}\n'.format(parsimony_coefficient))
        fp.write('Stopping Threshold: {}\n'.format(stopping_threshold))
        fp.write('Max Generations: {}\n'.format(max_generations))
        fp.write('Const Range: {}\n'.format(const_range))
        fp.write('Tournament Size: {}\n'.format(tournament_size))
        fp.write('Integer Constants: {}\n'.format(int_consts))
        fp.write('Crossover Probability: {}\n'.format(crossover_probability))
        fp.write('Mutation Probability: {}\n'.format(mutation_probability))
        fp.write('Clone Probability: {}\n'.format(clone_probability))
        fp.write('Init Depth: {}\n'.format(init_depth))
        fp.write('Include Analytic: {}\n'.format(include_analytic))
        fp.write('Fitness: {}\n'.format({0: 'MSE', 1: 'MAE', 2: 'R^2'}[fitness]))
        fp.write('Num Runs: {}\n\n\n'.format(num_runs))
        fp.write('*****RESULTS*****\nTest Set Scores:\n')
        for i, score in enumerate(test_scores):
            fp.write('Run {}: {:10.3f}\n'.format(i, score))


if __name__ == '__main__':
    main()
