
from typing_extensions import TypeAlias

from .classes import Generation, AlgorithmParams

from .geneticalgorithm2 import GeneticAlgorithm2

# to keep backward compatibility
geneticalgorithm2: TypeAlias = GeneticAlgorithm2

from .mutations import Mutations
from .crossovers import Crossover
from .selections import Selection

from .initializer import Population_initializer

from .cache import np_lru_cache
from .callbacks import Callbacks, Actions, ActionConditions, MiddleCallbacks

from .plotting_tools import plot_pop_scores, plot_several_lines




