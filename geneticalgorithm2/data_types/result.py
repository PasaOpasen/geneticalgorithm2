import warnings
from dataclasses import dataclass

from ..utils.aliases import array1D

from .generation import Generation
from .base import DictLikeGetSet


@dataclass
class GAResult(DictLikeGetSet):

    last_generation: Generation

    @property
    def variable(self) -> array1D:
        return self.last_generation.variables[0]

    @property
    def score(self) -> float:
        return self.last_generation.scores[0]

    @property
    def function(self):
        warnings.warn(
            f"'function' field is deprecated, will be removed in version 7, "
            f"use 'score' to get best population score"
        )
        return self.score
