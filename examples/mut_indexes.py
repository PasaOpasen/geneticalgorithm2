# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 20:18:39 2021

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import GeneticAlgorithm2 as ga
from geneticalgorithm2 import AlgorithmParams

def f(X):
    return np.sum(X)

dim = 7    
    
varbound = [(0,10)]*dim

model = ga(
    function = f,
    dimension=dim,
    variable_type='real',
    variable_boundaries=varbound,
    algorithm_parameters=AlgorithmParams(
        max_num_iteration=None,
        mutation_probability=0.2,
        elit_ratio=0.01,
        crossover_probability=0.5,
        parents_portion=0.3,
        crossover_type='uniform',
        mutation_type='uniform_by_center',
        selection_type='roulette',
        max_iteration_without_improv=None
    )
)

pop = np.full((10, dim), 5, dtype = np.float32)

model.run(no_plot = False,
          start_generation = (pop, None)
          )

print(model.result.last_generation.variables)


# freeze dims [0,1,2,3] for mutation -- they won't be changed
model.run(no_plot = False,
          start_generation = (pop, None),
          mutation_indexes= [6, 5, 4]
          )


print(model.result.last_generation.variables)

