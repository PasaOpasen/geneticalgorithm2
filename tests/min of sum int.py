# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:18:37 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np
from geneticalgorithm2 import GeneticAlgorithm2 as ga

def f(X):
    return np.sum(X)
    
    
varbound = [[0,10]]*3

model = ga(function=f, dimension=3, variable_type='int', variable_boundaries=varbound)

model.run()


# check discrete mutation

varbound = [[0, 10]] * 300

model = ga(
    dimension=300, variable_type='int',
    variable_boundaries=varbound,
    algorithm_parameters={
       'mutation_discrete_type': 'uniform_discrete',
       'max_num_iteration': 1000
    }
)

model.run(stop_when_reached=0, function=f,)


model = ga(function=f, dimension=300, variable_type='int',
           variable_boundaries=varbound,
           algorithm_parameters={
               'mutation_discrete_type': lambda x, left, right: left,
               'max_num_iteration': 1000
           })

model.run(stop_when_reached=0)


