"""
Genetic Algorithm (Elitist version) for Python3.8+

An implementation of elitist genetic algorithm for solving problems with
continuous, integers, or mixed variables.

repo path:       https://github.com/PasaOpasen/geneticalgorithm2

code docs path:  https://pasaopasen.github.io/geneticalgorithm2/
"""


from typing_extensions import TypeAlias

from .data_types.algorithm_params import AlgorithmParams
from .data_types.generation import Generation

from .geneticalgorithm2 import GeneticAlgorithm2

# to keep backward compatibility like it was since geneticalgorithm package
geneticalgorithm2: TypeAlias = GeneticAlgorithm2

from .mutations import Mutations
from .crossovers import Crossover
from .selections import Selection

from .population_initializer import get_population_initializer

# to keep backward compatibility
Population_initializer: TypeAlias = get_population_initializer

from .callbacks import Callbacks, Actions, ActionConditions, MiddleCallbacks

from .utils.cache import np_lru_cache
from .utils.plotting import plot_pop_scores, plot_several_lines




