


import sys
sys.path.append('..')

import numpy as np

from OppOpPopInit import set_seed
from geneticalgorithm2 import geneticalgorithm2 as ga, AlgorithmParams, plot_several_lines

set_seed(1)

def f(X):
    return np.sum(X) + np.abs(X).sum() - X.mean()

varbound = (
    (0.5, 1.5),
    (1, 100),
    (0, 1),
    (10, 13),
    (-10, 10),
    (-50, 10)
)

vartype = ('real', 'int', 'real', 'real', 'real', 'int')


reports = []
probs = [0.1, 0.3, 0.5, 0.8, 1]

for p in probs:

    arrs = []

    for i in range(15):
        model = ga(
            function=f, dimension=len(vartype),
            variable_type=vartype,
            variable_boundaries=varbound,
            algorithm_parameters=AlgorithmParams(
                crossover_probability=p,
                max_num_iteration=400,
                elit_ratio=0.01
            )
        )

        model.run(no_plot=True)
        arrs.append(model.report)

    reports.append(np.array(arrs).mean(axis=0))

plot_several_lines(
    reports,
    colors=['red', 'blue', 'green', 'black', 'yellow', 'orange'],
    labels=[f"prob = {v}" for v in probs],
    save_as='./output/sense_of_crossover_prob__no_sense.png',

    title='result of different crossover probs',
    ylabel='avg score'
)


