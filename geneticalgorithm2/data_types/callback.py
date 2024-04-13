from typing import Optional, List, Callable

from dataclasses import dataclass

from ..crossovers import CrossoverFunc

from ..mutations import MutationFloatFunc, MutationIntFunc
from ..selections import SelectionFunc
from ..utils.aliases import array2D, array1D

from .generation import Generation
from .base import DictLikeGetSet


@dataclass
class MiddleCallbackData(DictLikeGetSet):
    """
    data object using with middle callbacks
    """

    reason_to_stop: Optional[str]

    last_generation: Generation

    current_generation: int
    report_list: List[float]

    mutation_prob: float
    mutation_discrete_prob: float

    mutation: MutationFloatFunc
    mutation_discrete: MutationIntFunc
    crossover: CrossoverFunc
    selection: SelectionFunc

    current_stagnation: int
    max_stagnation: int

    parents_portion: float
    elit_ratio: float

    set_function: Callable[[array2D], array1D]
