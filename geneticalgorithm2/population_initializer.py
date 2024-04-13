
from typing import Callable, Optional, Tuple, Literal

import numpy as np

from .utils.aliases import TypeAlias, array1D, array2D


LOCAL_OPTIMIZATION_STEP_CASE: TypeAlias = Literal['before_select', 'after_select', 'never']
"""
When the local optimization (candidates enhancing) must be performed:
    * 'never' -- don't do local optimization
    * 'before_select' -- before selection best N objects 
        (example: do local optimization for 5N objects and select N best results)
    * 'after_select' -- do local optimization on best selected N objects
"""


def get_population_initializer(
    select_best_of: int = 4,
    local_optimization_step: LOCAL_OPTIMIZATION_STEP_CASE = 'never',
    local_optimizer: Optional[
        Callable[
            [array1D, float],
            Tuple[array1D, float]
        ]
    ] = None
) -> Tuple[int, Callable[[array2D, array1D], Tuple[array2D, array1D]]]:
    """
    Args:
        select_best_of: determines population size to select 1/select_best_of best part of start population.
            For example, for select_best_of = 4 and population_size = N there will be selected N best objects
                from 5N generated objects (if start_generation=None dictionary).
            If start_generation is not None dictionary, there will be selected best (start_generation) / N objects
        local_optimization_step: when to perform local optimization
        local_optimizer: the local optimization function (object array, its score) -> (modified array, its score)

    Returns:
        select_best_of, function which will perform the selection and local optimization
    """
    
    assert select_best_of > 0 and isinstance(select_best_of, int), (select_best_of, type(select_best_of))

    assert local_optimization_step in LOCAL_OPTIMIZATION_STEP_CASE.__args__, (
        local_optimization_step, LOCAL_OPTIMIZATION_STEP_CASE.__args__
    )

    if local_optimizer is None and local_optimization_step in LOCAL_OPTIMIZATION_STEP_CASE.__args__[:2]:
        raise Exception(
            f"for local_optimization_step from {LOCAL_OPTIMIZATION_STEP_CASE.__args__[:2]} "
            f"local_optimizer function mustn't be None"
        )

    def select_best(population: array2D, scores: array1D) -> Tuple[array2D, array1D]:
        args = np.argsort(scores)
        args = args[:round(args.size/select_best_of)]
        return population[args], scores[args]

    def local_opt(population: array2D, scores: array1D):
        _pop, _score = zip(
            *[
                local_optimizer(population[i], scores[i]) for i in range(scores.size)
            ]
        )
        return np.array(_pop), np.array(_score)

    #def Create_population(func, start_generation, expected_size, #variable_boundaries):
    #    
    #    if not (start_generation['variables'] is None):
    #        pop = start_generation['variables']
    #        scores = start_generation['scores']
    #        if scores is None:
    #            scores = np.array([func(pop[i, :]) for i in range(pop.shape[0])])
    #        return pop, scores

    def process_population(population: array2D, scores: array1D):
        if local_optimization_step == 'before_select':
            pop, s = local_opt(population, scores)
            return select_best(pop, s)

        if local_optimization_step == 'after_select':
            pop, s = select_best(population, scores)
            return local_opt(pop, s)

        #if local_optimization_step == 'never':
        return select_best(population, scores)

    return select_best_of, process_population








