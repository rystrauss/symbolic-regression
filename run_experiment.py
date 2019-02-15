import json
import os
import pickle

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from srlearn import SymbolicRegressor
from srlearn.fitness import mean_absolute_error
from srlearn.functions import *


@click.command()
@click.argument('data_path', type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=1)
@click.argument('save_dir', type=click.Path(exists=False, file_okay=False, dir_okay=True), nargs=1)
@click.option('--population_size', type=click.INT, nargs=1, default=500)
@click.option('--parsimony_coefficient', type=click.FLOAT, nargs=1, default=0.001)
@click.option('--stopping_threshold', type=click.FLOAT, nargs=1, default=0.001)
@click.option('--max_generations', type=click.INT, nargs=1, default=50)
@click.option('--const_range', type=click.INT, nargs=2, default=(-5, 5))
@click.option('--tournament_size', type=click.INT, nargs=1, default=5)
@click.option('--int_consts', type=click.BOOL, nargs=1, default=False)
@click.option('--crossover_probability', type=click.FLOAT, nargs=1, default=0.9)
@click.option('--mutation_probability', type=click.FLOAT, nargs=1, default=0.05)
@click.option('--clone_probability', type=click.FLOAT, nargs=1, default=0.05)
@click.option('--init_depth', type=click.INT, nargs=1, default=6)
@click.option('--include_analytical', type=click.BOOL, nargs=1, default=False)
@click.option('--num_runs', type=click.INT, nargs=1, default=1)
def main(data_path, save_dir, population_size, parsimony_coefficient, stopping_threshold, max_generations, const_range,
         tournament_size, int_consts, crossover_probability, mutation_probability, clone_probability, init_depth,
         include_analytical, num_runs):
    data = pd.read_csv(data_path)
    X = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1])

    function_set = (add, subtract, multiply, divide, exp, cos, sin, log, tan, sqrt) if include_analytical else (
        add, subtract, multiply, divide)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

    test_scores = []

    for i in range(num_runs):
        print('Starting run {}...'.format(i))
        model = SymbolicRegressor(population_size=population_size,
                                  fitness_function=mean_absolute_error,
                                  function_set=function_set,
                                  parsimony_coefficient=parsimony_coefficient,
                                  const_range=const_range,
                                  tournament_size=tournament_size,
                                  int_consts=int_consts,
                                  crossover_probability=crossover_probability,
                                  mutation_probability=mutation_probability,
                                  clone_probability=clone_probability,
                                  init_depth=init_depth)

        history = model.fit(X_train, y_train, stopping_threshold=stopping_threshold, max_generations=max_generations)

        test_scores.append(model.score(X_test, y_test))

        if not os.path.exists(os.path.join(save_dir, 'run_{}'.format(i))):
            os.makedirs(os.path.join(save_dir, 'run_{}'.format(i)))

        with open(os.path.join(save_dir, 'run_{}'.format(i), 'history.json'), 'w') as fp:
            json.dump(history, fp)

        with open(os.path.join(save_dir, 'run_{}'.format(i), 'model.p'), 'wb') as fp:
            pickle.dump(model, fp)

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
        fp.write('Include Analytical: {}\n'.format(include_analytical))
        fp.write('Num Runs: {}\n\n\n'.format(num_runs))
        fp.write('*****RESULTS*****\nTest Set Scores:\n')
        for i, score in enumerate(test_scores):
            fp.write('Run {}: {:10.3f}\n'.format(i, score))


if __name__ == '__main__':
    main()
