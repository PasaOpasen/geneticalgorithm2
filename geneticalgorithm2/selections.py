
from typing import Callable, Dict, List

import math
import random
import numpy as np

from .utils.aliases import array1D, TypeAlias

SelectionFunc: TypeAlias = Callable[[array1D, int], array1D]
"""
Function (scores, count to select) -> indexes of selected
"""


#region UTILS

def inverse_scores(scores: array1D) -> array1D:
    """
    inverses scores (min values become to max)
    """
    minobj = scores[0]
    normobj = scores - minobj if minobj < 0 else scores

    return (np.amax(normobj) + 1) - normobj


def roulette(scores: array1D, parents_count: int) -> array1D:
    """simplest roulette selector for which the highest score means more preferred"""

    sum_normobj = np.sum(scores)
    prob = scores / sum_normobj
    cumprob = np.cumsum(prob)

    parents_indexes = np.array(
        [
            index if index < cumprob.size else np.random.randint(0, index - 1)
            for index in (
                np.searchsorted(cumprob, np.random.random())
                for _ in range(parents_count)
            )
        ]
    )

    return parents_indexes


#endregion

class Selection:
    """
    Selections functions static class
    
    Selection function selects the population subset according to scores and its own rules
    """

    @staticmethod
    def selections_dict() -> Dict[str, SelectionFunc]:
        return {
            n: getattr(Selection, n)()
            for n in (
                'fully_random',
                'roulette',
                'stochastic',
                'sigma_scaling',
                'ranking',
                'linear_ranking',
                'tournament',
            )
        }

    @staticmethod
    def fully_random() -> SelectionFunc:
        """returns the selector of fully random parents (for tests purposes)"""
        
        def func(scores: array1D, parents_count: int):
            indexes = np.arange(parents_count)
            return np.random.choice(indexes, parents_count, replace=False)
        
        return func

    @staticmethod
    def roulette() -> SelectionFunc:
        
        def func(scores: array1D, parents_count: int):
            return roulette(inverse_scores(scores), parents_count)
        
        return func

    @staticmethod
    def stochastic() -> SelectionFunc:
        
        def func(scores: np.ndarray, parents_count: int):
            f = inverse_scores(scores)
            
            fN: float = 1.0 / parents_count
            k: int = 0
            acc: float = 0.0
            parents: List[int] = []
            r: float = random.random() * fN
            
            while len(parents) < parents_count:
                
                acc += f[k]
                
                while acc > r:
                    parents.append(k)
                    if len(parents) == parents_count: 
                        break
                    r += fN
                
                k += 1
            
            return np.array(parents[:parents_count])
        
        return func

    @staticmethod
    def sigma_scaling(epsilon: float = 0.01, is_noisy: bool = False) -> SelectionFunc:
        
        def func(scores: array1D, parents_count):
            f = inverse_scores(scores)
            
            sigma = np.std(f, ddof = 1) if is_noisy else np.std(f)
            average = np.mean(f)
            
            if sigma == 0:
                f = 1
            else:
                f = np.maximum(epsilon, 1 + (f - average)/(2*sigma))
            
            return roulette(f, parents_count)
        
        return func

    @staticmethod
    def ranking() -> SelectionFunc:
        
        def func(scores: array1D, parents_count: int):
            return roulette(1 + np.arange(parents_count)[::-1], parents_count)
        
        return func

    @staticmethod
    def linear_ranking(selection_pressure: float = 1.5) -> SelectionFunc:
        
        assert 1 <= selection_pressure <= 2, f"selection_pressure should be in (1, 2), but got {selection_pressure}"
        
        def func(scores: array1D, parents_count: int):
            tmp = parents_count * (parents_count-1)
            alpha = (2 * parents_count - selection_pressure * (parents_count + 1)) / tmp
            beta = 2 * (selection_pressure - 1) / tmp

            a = -2 * alpha - beta
            b = (2 * alpha + beta) ** 2
            c = 8 * beta
            d = 2 * beta
            
            indexes = np.arange(parents_count)
            
            return np.array(
                [
                    indexes[-round((a + math.sqrt(b + c * random.random())) / d)]
                    for _ in range(parents_count)
                ]
            )

        return func

    @staticmethod
    def tournament(tau: int = 2) -> SelectionFunc:

        # NOTE
        # this code really does tournament selection
        # because scores are always sorted
        
        def func(scores: array1D, parents_count: int):
            
            indexes = np.arange(parents_count)
            
            return np.array(
                [
                    np.min(np.random.choice(indexes, tau, replace=False))
                    for _ in range(parents_count)
                ]
            )
        
        return func
    
    
    

















