# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 15:22:52 2021

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import AlgorithmParams
from geneticalgorithm2 import GeneticAlgorithm2 as ga

def function(X):
    return np.sum(X)
    
    
var_bound = np.array([[0,10]]*3)


model = ga(function, dimension = 3, 
                variable_type='real', 
                 variable_boundaries = var_bound,
                 variable_type_mixed = None, 
                 function_timeout = 10,
                 algorithm_parameters={'max_num_iteration': None,
                                       'population_size':100,
                                       'mutation_probability':0.1,
                                       'elit_ratio': 0.01,
                                       'crossover_probability': 0.5,
                                       'parents_portion': 0.3,
                                       'crossover_type':'uniform',
                                       'mutation_type': 'uniform_by_center',
                                       'selection_type': 'roulette',
                                       'max_iteration_without_improv':None}
            )

model.run(no_plot = False)


# from version 6.3.0 it is equal to

model = ga(function, dimension = 3, 
                variable_type='real', 
                 variable_boundaries = var_bound,
                 variable_type_mixed = None, 
                 function_timeout = 10,
                 algorithm_parameters=AlgorithmParams(
                     max_num_iteration = None,
                     population_size = 100,
                     mutation_probability = 0.1,
                     elit_ratio = 0.01,
                     crossover_probability = 0.5,
                     parents_portion = 0.3,
                     crossover_type = 'uniform',
                     mutation_type = 'uniform_by_center',
                     selection_type = 'roulette',
                     max_iteration_without_improv = None
                     )
            )


model.run(no_plot = False)


# or

model = ga(function, dimension = 3, 
                variable_type='real', 
                 variable_boundaries = var_bound,
                 variable_type_mixed = None, 
                 function_timeout = 10,
                 algorithm_parameters=AlgorithmParams( )
            )


model.run(no_plot = False)

















