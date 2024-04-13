
import sys
sys.path.append('..')

import numpy as np
from geneticalgorithm2 import GeneticAlgorithm2 as ga

def f(X):
    return np.sum(X)

varbound = (
    (0.5, 1.5),
    (1, 100),
    (-100, 1)
)

vartype = ('real', 'real', 'int')

model = ga(
    function=f, dimension=len(vartype),
    variable_type=vartype,
    variable_boundaries=varbound
)

# old!!
model.run(disable_progress_bar=True)


model.run(progress_bar_stream=None)
model.run(progress_bar_stream='stdout')
model.run(progress_bar_stream='stderr')
