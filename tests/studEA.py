# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 02:44:17 2020

@author: qtckp
"""

import sys
sys.path.append('..')


import numpy as np
from geneticalgorithm2 import GeneticAlgorithm2 as ga
import matplotlib.pyplot as plt

def f(X):
    return np.sum(X)
    
    
varbound = [[0,10]]*20

start_gen = np.random.uniform(0, 10, (100, 20))

model = ga(function=f, dimension=20, variable_type='real', 
           variable_boundaries=varbound,
           algorithm_parameters={
                   'max_num_iteration': 400
                   })



for stud in (False, True):
    
    model.run(no_plot = True, studEA= stud, start_generation=(start_gen, None), seed=1)
    
    plt.plot(model.report, label = f"studEA strategy = {stud}")
    
    
    
plt.xlabel('Generation')
plt.ylabel('Minimized function')
plt.title('Using stud EA strategy')
plt.legend()


plt.savefig("./output/studEA.png", dpi = 300)
plt.show()
