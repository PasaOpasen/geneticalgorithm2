
from typing import Callable, Optional, Tuple, Literal

import numpy as np

from .utils.aliases import TypeAlias, array1D, array2D


LOCAL_OPT_STEPS: TypeAlias = Literal['before_select', 'after_select', 'never']


def Population_initializer(
    select_best_of: int = 4,
    local_optimization_step: LOCAL_OPT_STEPS = 'never',
    local_optimizer: Optional[
        Callable[
            [array1D, float],
            Tuple[array1D, float]
        ]
    ] = None
) -> Tuple[int, Callable[[array2D, array1D], Tuple[array2D, array1D]]]:
    """
    select_best_of (int) -- select 1/select_best_of best part of start population. For example, for select_best_of = 4 and population_size = N will be selected N best objects from 5N generated objects (if start_generation = None dictionary). If start_generation is not None dictionary, it will be selected best size(start_generation)/N  objects

    local_optimization_step (str) -- when should we do local optimization? Available values:
    
    * 'never' -- don't do local optimization
    * 'before_select' -- before selection best N objects (example: do local optimization for 5N objects and select N best results)
    * 'after_select' -- do local optimization on best selected N objects

    local_optimizer (function) -- local optimization function like:
        def loc_opt(object_as_array, current_score):
            # some code
            return better_object_as_array, better_score
    """
    
    assert (select_best_of > 0 and type(select_best_of) == int), "select_best_of argument should be integer and more than 0"

    assert (local_optimization_step in LOCAL_OPT_STEPS.__args__), f"local_optimization_step should be in {LOCAL_OPT_STEPS.__args__}, but got {local_optimization_step}"

    if local_optimizer is None and local_optimization_step in LOCAL_OPT_STEPS.__args__[:2]:
        raise Exception(f"for local_optimization_step from {LOCAL_OPT_STEPS.__args__[:2]} local_optimizer function mustn't be None")


    def Select_best(population: array2D, scores: array1D) -> Tuple[array2D, array1D]:
        args = np.argsort(scores)
        args = args[:round(args.size/select_best_of)]
        return population[args], scores[args]

    def Local_opt(population: array2D, scores: array1D):
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


    def Result(population: array2D, scores: array1D):
        if local_optimization_step == 'before_select':
            pop, s = Local_opt(population, scores)
            return Select_best(pop, s)

        if local_optimization_step == 'after_select':
            pop, s = Select_best(population, scores)
            return Local_opt(pop, s)

        #if local_optimization_step == 'never':
        return Select_best(population, scores)
    

    return select_best_of, Result








