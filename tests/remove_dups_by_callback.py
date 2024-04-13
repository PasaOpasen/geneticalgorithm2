# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 14:15:51 2021

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import GeneticAlgorithm2 as ga

from geneticalgorithm2 import ActionConditions, Actions, MiddleCallbacks

from OppOpPopInit import OppositionOperators, SampleInitializers

np.random.seed(1)

def converter(arr):
    arrc = np.full_like(arr, 3)
    arrc[arr < 0.75 ] = 2
    arrc[arr < 0.50 ] = 1
    arrc[arr < 0.25 ] = 0
    return arrc

def f(X):
    return np.sum(converter(X))

dim = 60
    
varbound = np.array([[0, 1]]*dim)
start_pop = SampleInitializers.CreateSamples(
    SampleInitializers.Uniform(minimums=varbound[:, 0], maximums=varbound[:, 1]),
    count=50
)

model = ga(function=f, dimension = dim,
           variable_type='real',
           variable_boundaries=varbound,
           algorithm_parameters = {
                'max_num_iteration': 1000,
                'mutation_probability':0.1,
                'elit_ratio': 0.01,
                'crossover_probability': 0.5,
                'parents_portion': 0.3,
                'crossover_type':'uniform',
                'mutation_type': 'uniform_by_center',
                'selection_type': 'roulette',
                'max_iteration_without_improv': None
           }
)

model.run(
    no_plot = False,
    stop_when_reached = 0,
    start_generation=(start_pop, None)
)
model.plot_generation_scores(title = 'Population scores after ending of searching', save_as= './output/with_dups.png')

model.run(
    no_plot = False,
    stop_when_reached = 0,
    start_generation=(start_pop, None),
    middle_callbacks = [
        MiddleCallbacks.UniversalCallback(
            Actions.RemoveDuplicates(
                oppositor = OppositionOperators.Continual.abs(
                    minimums = varbound[:, 0],
                    maximums = varbound[:, 1]
                ),
                converter = converter
            ),
        ActionConditions.EachGen(15))
    ]
          
)

model.plot_generation_scores(title = 'Population scores after ending of searching', save_as= './output/without_dups.png')





