
from typing import Callable, Dict, Union

import random
import numpy as np


from .utils import fast_min, fast_max

from .aliases import TypeAlias

MutationFloatFunc: TypeAlias = Callable[[float, float, float], float]
MutationIntFunc: TypeAlias = Callable[[int, int, int], int]
MutationFunc: TypeAlias = Union[MutationIntFunc, MutationFloatFunc]


class Mutations:

    @staticmethod
    def mutations_dict() -> Dict[str, MutationFloatFunc]:
        return {
            'uniform_by_x': Mutations.uniform_by_x(),
            'uniform_by_center': Mutations.uniform_by_center(),
            'gauss_by_center': Mutations.gauss_by_center(),
            'gauss_by_x': Mutations.gauss_by_x(),
        }

    @staticmethod
    def mutations_discrete_dict() -> Dict[str, MutationIntFunc]:
        return {
            'uniform_discrete': Mutations.uniform_discrete()
        }


    @staticmethod
    def uniform_by_x() -> MutationFloatFunc:
        
        def func(x: float, left: float, right: float):
            alp: float = fast_min(x - left, right - x)
            return random.uniform(x - alp, x + alp)
        return func

    @staticmethod
    def uniform_by_center() -> MutationFloatFunc:
        
        def func(x: float, left: float, right: float):
            return random.uniform(left, right)
        
        return func

    @staticmethod
    def gauss_by_x(sd: float = 0.3) -> MutationFloatFunc:
        """
        gauss mutation with x as center and sd*length_of_zone as std
        """
        def func(x: float, left: float, right: float):
            std: float = sd * (right - left)
            return fast_max(
                left, 
                fast_min(right, np.random.normal(loc=x, scale=std))
            )
        
        return func

    @staticmethod
    def gauss_by_center(sd: float = 0.3) -> MutationFloatFunc:
        """
        gauss mutation with (left+right)/2 as center and sd*length_of_zone as std
        """
        def func(x: float, left: float, right: float):
            std: float = sd * (right - left)
            return fast_max(
                left, 
                fast_min(right, np.random.normal(loc=(left + right) * 0.5, scale=std))
            )
        
        return func

    @staticmethod
    def uniform_discrete() -> MutationIntFunc:
        def func(x: int, left: int, right: int) -> int:
            return random.randint(left, right)
        return func

