
import math
import numpy as np
from geneticalgorithm2 import GeneticAlgorithm2 as ga


def f_slow(X):
    """
    slow function
    """
    a = X[0]
    b = X[1]
    c = X[2]
    s = 0
    for i in range(10000):
        s += math.sin(a * i) + math.sin(b * i) + math.cos(c * i)

    return s

rg = np.arange(10000)
def f_fast(X):
    """
    fast function
    """
    a, b, c = X
    return (np.sin(rg*a) + np.sin(rg*b) + np.cos(rg * c)).sum()


algorithm_param = {'max_num_iteration': 50,
                   'population_size': 100,
                   'mutation_probability': 0.1,
                   'elit_ratio': 0.01,
                   'parents_portion': 0.3,
                   'crossover_type': 'uniform',
                   'mutation_type': 'uniform_by_center',
                   'selection_type': 'roulette',
                   'max_iteration_without_improv': None}

varbound = [(-10, 10)] * 3

model = ga(function=f_slow, dimension=3,
           variable_type='real',
           variable_boundaries=varbound,
           algorithm_parameters=algorithm_param)

######## compare parallel and normal with slow function

%time model.run()
# Wall time: 34.7s

%time model.run(set_function=ga.set_function_multiprocess(f_slow, n_jobs=3))
# Wall time: 23 s


######## compare default and vectorized on fast func and small pop

model = ga(function=f_fast, dimension=3,
           variable_type='real',
           variable_boundaries=varbound,
           algorithm_parameters=algorithm_param)

%timeit model.run(set_function=ga.default_set_function(f_fast), no_plot=True, progress_bar_stream=None, disable_printing=True)
# 1.41 s ± 4.79 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit model.run(set_function=ga.vectorized_set_function(f_fast), no_plot=True, progress_bar_stream=None, disable_printing=True)
# 1.42 s ± 10.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

######## compare default and vectorized on fast func and big pop
algorithm_param['population_size'] = 1500
algorithm_param['max_num_iteration'] = 15
model = ga(function=f_fast, dimension=3,
           variable_type='real',
           variable_boundaries=varbound,
           algorithm_parameters=algorithm_param)

%timeit model.run(set_function=ga.default_set_function(f_fast), no_plot=True, progress_bar_stream=None, disable_printing=True)
# 6.63 s ± 229 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit model.run(set_function=ga.vectorized_set_function(f_fast), no_plot=True, progress_bar_stream=None, disable_printing=True)
# 6.47 s ± 87.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


