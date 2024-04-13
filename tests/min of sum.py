# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:15:47 2020

@author: qtckp
"""
import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga

def f(X):
    return np.sum(X)
    
    
varbound = [[0,10]]*20

model = ga(function=f, dimension=20, variable_type='real', variable_boundaries=varbound)

model.run(no_plot = False)