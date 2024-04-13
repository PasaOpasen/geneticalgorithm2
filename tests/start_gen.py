# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 00:18:17 2020

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga

def f(X):
    return np.sum(X)
    
dim = 6
    
varbound = [(0,10)]*dim


algorithm_param = {'max_num_iteration': 500,
                   'population_size':100,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None}

model = ga(function=f, 
           dimension=dim, 
           variable_type='real', 
           variable_boundaries=varbound,
           algorithm_parameters = algorithm_param)

# start generation
# as u see u can use any values been valid for ur function
samples = np.random.uniform(0, 50, (300, dim)) # 300 is the new size of your generation



model.run(no_plot = False, start_generation={'variables':samples, 'scores': None})
# it's not necessary to evaluate scores before
# but u can do it if u have evaluated scores and don't wanna repeat calcucations


# from version 6.3.0 it's recommended to use this form
from geneticalgorithm2 import Generation
model.run(no_plot = False, start_generation=Generation(variables = samples, scores = None))


# from version 6.4.0 u also can use these forms
model.run(no_plot = False, start_generation= samples)
model.run(no_plot = False, start_generation= (samples, None))


# if u have scores array, u can put it too
scores = np.array([f(sample) for sample in samples])
model.run(no_plot = False, start_generation= (samples, scores))


# okay, let's continue optimization using saved last generation

model.run(no_plot = False, start_generation=model.result.last_generation)
