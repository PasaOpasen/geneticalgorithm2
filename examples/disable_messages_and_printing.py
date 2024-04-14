# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:46:10 2021

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import GeneticAlgorithm2 as ga

def f(X):
    return np.sum(X)
    
    
varbound = [[0,30]]*20

model = ga(function=f, dimension=20, variable_type='real', variable_boundaries=varbound)

result = model.run(
    no_plot = True,
    progress_bar_stream=None,
    disable_printing=True
)

print(result.function)