
from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Literal


from dataclasses import dataclass
import warnings

from .base import DictLikeGetSet
from ..utils.funcs import can_be_prob

from ..crossovers import Crossover, CrossoverFunc
from ..mutations import Mutations, MutationIntFunc, MutationFloatFunc
from ..selections import Selection, SelectionFunc


_algorithm_params_slots = {
    'max_num_iteration',
    'max_iteration_without_improv',
    'population_size',
    'mutation_probability',
    'mutation_discrete_probability',
    'elit_ratio',
    'crossover_probability',
    'parents_portion',
    'crossover_type',
    'mutation_type',
    'mutation_discrete_type',
    'selection_type'
}


@dataclass
class AlgorithmParams(DictLikeGetSet):

    max_num_iteration: Optional[int] = None
    max_iteration_without_improv: Optional[int] = None

    population_size: int = 100

    mutation_probability: float = 0.1
    mutation_discrete_probability: Optional[float] = None

    #  deprecated
    crossover_probability: Optional[float] = None

    elit_ratio: float = 0.04
    parents_portion: float = 0.3

    crossover_type: Union[str, CrossoverFunc] = 'uniform'
    mutation_type: Union[str, MutationFloatFunc] = 'uniform_by_center'
    mutation_discrete_type: Union[str, MutationIntFunc] = 'uniform_discrete'
    selection_type: Union[str, SelectionFunc] = 'roulette'

    def validate(self) -> None:

        assert int(self.population_size) > 0, f"population size must be integer and >0, not {self.population_size}"
        assert (can_be_prob(self.parents_portion)), "parents_portion must be in range [0,1]"
        assert (can_be_prob(self.mutation_probability)), "mutation_probability must be in range [0,1]"
        assert (can_be_prob(self.elit_ratio)), "elit_ratio must be in range [0,1]"

        if self.max_iteration_without_improv is not None and self.max_iteration_without_improv < 1:
            warnings.warn(
                f"max_iteration_without_improv is {self.max_iteration_without_improv} but must be None or int > 0"
            )
            self.max_iteration_without_improv = None

    def get_CMS_funcs(self) -> Tuple[
        CrossoverFunc,
        MutationFloatFunc,
        MutationIntFunc,
        SelectionFunc
    ]:
        """
        returns gotten crossover, mutation, discrete mutation, selection
        as necessary functions
        """

        result: List[Callable] = []
        for name, value, dct in (
            ('crossover', self.crossover_type, Crossover.crossovers_dict()),
            ('mutation', self.mutation_type, Mutations.mutations_dict()),
            ('mutation_discrete', self.mutation_discrete_type, Mutations.mutations_discrete_dict()),
            ('selection', self.selection_type, Selection.selections_dict())
        ):
            if isinstance(value, str):
                if value not in dct:
                    raise ValueError(
                        f"unknown name of {name}: '{value}', must be from {tuple(dct.keys())} or a custom function"
                    )
                result.append(dct[value])
            else:
                assert callable(value), f"{name} must be string or callable"
                result.append(value)

        return tuple(result)

    @staticmethod
    def from_dict(dct: Dict[str, Any]):

        result = AlgorithmParams()

        for name, value in dct.items():
            if name not in _algorithm_params_slots:
                raise AttributeError(f"name '{name}' does not exists in AlgorithmParams fields")

            setattr(result, name, value)
        return result













