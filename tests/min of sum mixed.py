# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:20:05 2020

@author: qtckp
"""

import sys
sys.path.append('..')

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga

def f(X):
    return np.sum(X)

varbound = (
    (0.5, 1.5),
    (1, 100),
    (0, 1)
)

vartype = ('real', 'int', 'int')

model = ga(
    function=f, dimension=len(vartype),
    variable_type=vartype,
    variable_boundaries=varbound
)

model.run()


