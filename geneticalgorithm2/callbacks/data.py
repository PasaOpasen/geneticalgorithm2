from typing import Optional, List, Callable, Tuple

from dataclasses import dataclass
from typing_extensions import TypeAlias

from ..crossovers import CrossoverFunc

from ..mutations import MutationFloatFunc, MutationIntFunc
from ..selections import SelectionFunc
from ..utils.aliases import array2D, array1D

from ..data_types.aliases import SetFunctionToMinimize
from ..data_types.generation import Generation
from ..data_types.base import DictLikeGetSet


@dataclass
class MiddleCallbackData(DictLikeGetSet):
    """
    data object using in middle callbacks
    """

    # TODO use slots?

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

    set_function: SetFunctionToMinimize


SimpleCallbackFunc: TypeAlias = Callable[[int, List[float], array2D, array1D], None]
"""
Callback function performs any operations on 
    (generation number, best scores report list, last population matrix, last scores vector)

Notes: generation number cannot be changed
"""

MiddleCallbackConditionFunc: TypeAlias = Callable[[MiddleCallbackData], bool]
"""Function (middle callback data) -> (bool value means whether to call middle callback action)"""


MiddleCallbackActionFunc: TypeAlias = Callable[[MiddleCallbackData], MiddleCallbackData]
"""Function which transforms and returns middle callback data or just uses it some way"""


MiddleCallbackFunc: TypeAlias = Callable[[MiddleCallbackData], Tuple[MiddleCallbackData, bool]]
"""
Function (input middle callback data) -> (output callback data, changes flag)
    where input and output data may be same 
    and changes flag means whether the output data must be read back
        to the optimization process (to update by flag only one time -- for acceleration purposes)
"""