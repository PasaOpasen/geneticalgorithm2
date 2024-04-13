
from typing import Callable
from typing_extensions import TypeAlias

from ..utils.aliases import array2D, array1D


FunctionToMinimize: TypeAlias = Callable[[array1D], float]
"""usual (vector -> value) function to minimize"""


SetFunctionToMinimize: TypeAlias = Callable[[array2D], array1D]
"""
(population -> scores) function to minimize

it is like a vectorized version of usual (vector -> value) function
    performing to all population samples in the one call

but it can be written in more optimal way to speed up the calculations;
    also it can contain any logic due to samples relations and so on -- depends on the task 
"""

