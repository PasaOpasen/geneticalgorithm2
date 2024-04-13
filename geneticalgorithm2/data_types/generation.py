
from typing import Union, Dict, Literal, Tuple, Optional
from typing_extensions import TypeAlias

import os.path
from dataclasses import dataclass

import numpy as np

from ..utils.aliases import array2D, array1D, PathLike
from ..utils.files import mkdir_of_file
from ..utils.funcs import union_to_matrix

from .base import DictLikeGetSet

GenerationConvertible: TypeAlias = Union[
    'Generation',
    str,
    Dict[Literal['population', 'scores'], Union[array2D, array1D]],
    array2D,
    Tuple[
        Optional[array2D],
        Optional[array1D]
    ]
]
"""
The forms convertible to `Generation` object:
    - `Generation` object
    - path to saved generation
    - dict {'population': pop_matrix, 'scores': scores_vector}
    - wide population matrix
    - pair (pop_matrix, scores_vector)
"""


@dataclass
class Generation(DictLikeGetSet):
    """wrapper on generation object (pair of samples matrix and samples scores vector)"""
    variables: Optional[array2D] = None
    scores: Optional[array1D] = None

    def _check_dims(self) -> None:
        if self.variables is not None:
            assert len(self.variables.shape) == 2, (
                f"'variables' must be matrix with shape (objects, dimensions), not {self.variables.shape}"
            )
            if self.scores is not None:
                assert len(self.scores.shape) == 1, f"'scores' must be 1D-array, not with shape {self.scores.shape}"
                assert self.variables.shape[0] == self.scores.size, (
                    f"count of objects ({self.variables.shape[0]}) "
                    f"must be equal to count of scores ({self.scores.size})"
                )

    @property
    def size(self) -> int:
        return self.scores.size

    @property
    def dim_size(self) -> int:
        return self.variables.shape[1]

    def as_wide_matrix(self) -> array2D:
        # should not be used in main code -- was needed for old versions
        return union_to_matrix(self.variables, self.scores)

    def save(self, path: PathLike):
        mkdir_of_file(path)
        np.savez(path, population=self.variables, scores=self.scores)

    @staticmethod
    def load(path: PathLike):
        try:
            st = np.load(path)
        except Exception as err:
            raise Exception(
                f"if generation object is a string, "
                f"it must be path to npz file with needed content, but raised exception {repr(err)}"
            )

        assert 'population' in st and 'scores' in st, (
            "saved generation object must contain 'population' and 'scores' fields"
        )

        return Generation(variables=st['population'], scores=st['scores'])

    @staticmethod
    def from_object(
        dim: int,
        obj: GenerationConvertible
    ):
        """class constructor"""

        if isinstance(obj, np.ndarray):

            assert len(obj.shape) == 2 and (obj.shape[1] == dim or obj.shape[1] == dim + 1), (
                f"if start_generation is numpy array, "
                f"it must be with shape (samples, dim) or (samples, dim+1), not {obj.shape}"
            )

            generation = Generation(obj, None) if obj.shape[1] == dim else Generation.from_pop_matrix(obj)

        elif isinstance(obj, tuple):

            assert len(obj) == 2, (
                f"if start_generation is tuple, "
                f"it must be tuple with 2 components, not {len(obj)}"
            )

            variables, scores = obj

            assert (variables is None or scores is None) or (variables.shape[0] == scores.size), (
                "start_generation object must contain variables and scores components "
                "which are None or 2D- and 1D-arrays with same shape"
            )

            generation = Generation(variables=variables, scores=scores)

        elif isinstance(obj, dict):
            assert (
                ('variables' in obj and 'scores' in obj) and
                (
                    (obj['variables'] is None or obj['scores'] is None) or
                    (obj['variables'].shape[0] == obj['scores'].size)
                )
            ), (
                "start_generation object must contain 'variables' and 'scores' keys "
                "which are None or 2D- and 1D-arrays with same shape"
            )

            generation = Generation(variables=obj['variables'], scores=obj['scores'])

        elif isinstance(obj, Generation):
            generation = Generation(variables=obj['variables'], scores=obj['scores'])
        else:

            path = str(obj)
            if not os.path.exists(path):
                raise TypeError(
                    f"invalid type of generation! "
                    f"Must be in (Union[str, Dict[str, np.ndarray], Generation, np.ndarray, "
                    f"Tuple[Optional[np.ndarray], Optional[np.ndarray]]]), "
                    f"not {type(obj)}"
                )
            generation = Generation.load(path)

        generation._check_dims()

        if generation.variables is not None:
            assert generation.dim_size == dim, (
                f"generation dimension size {generation.dim_size} does not equal to target size {dim}"
            )

        return generation

    @staticmethod
    def from_pop_matrix(pop: array2D):
        import warnings
        warnings.warn(
            "depricated! pop matrix style will be removed at version 7, "
            "use samples and scores separetly"
        )
        return Generation(
            variables=pop[:, :-1],
            scores=pop[:, -1]
        )


