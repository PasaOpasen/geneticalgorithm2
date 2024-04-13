
from typing import Callable, Tuple, Dict

import random
import numpy as np

from .utils.aliases import TypeAlias, array1D

CrossoverFunc: TypeAlias = Callable[[array1D, array1D], Tuple[array1D, array1D]]


def get_copies(x: array1D, y: array1D) -> Tuple[array1D, array1D]:
    return x.copy(), y.copy()


class Crossover:

    @staticmethod
    def crossovers_dict() -> Dict[str, CrossoverFunc]:
        return {
            'one_point': Crossover.one_point(),
            'two_point': Crossover.two_point(),
            'uniform': Crossover.uniform(),
            'segment': Crossover.segment(),
            'shuffle': Crossover.shuffle(),
        }
    
    @staticmethod
    def one_point() -> CrossoverFunc:
        
        def func(x: array1D, y: array1D):
            ofs1, ofs2 = get_copies(x, y)
        
            ran = np.random.randint(0, x.size)
            
            ofs1[:ran] = y[:ran]
            ofs2[:ran] = x[:ran]
            
            return ofs1, ofs2
        return func
    
    @staticmethod
    def two_point() -> CrossoverFunc:
        
        def func(x: array1D, y: array1D):
            ofs1, ofs2 = get_copies(x, y)
        
            ran1 = np.random.randint(0, x.size)
            ran2 = np.random.randint(ran1, x.size)
            
            ofs1[ran1:ran2] = y[ran1:ran2]
            ofs2[ran1:ran2] = x[ran1:ran2]

            return ofs1, ofs2
        return func
    
    @staticmethod
    def uniform() -> CrossoverFunc:
        
        def func(x: array1D, y: array1D):
            ofs1, ofs2 = get_copies(x, y)
        
            ran = np.random.random(x.size) < 0.5
            ofs1[ran] = y[ran]
            ofs2[ran] = x[ran]
              
            return ofs1, ofs2
        
        return func
    
    @staticmethod
    def segment(prob: int = 0.6) -> CrossoverFunc:
        
        def func(x: array1D, y: array1D):
            
            ofs1, ofs2 = get_copies(x, y)
            
            p = np.random.random(x.size) < prob
            
            for i, val in enumerate(p):
                if val:
                    ofs1[i], ofs2[i] = ofs2[i], ofs1[i]
            
            return ofs1, ofs2
        
        return func
    
    @staticmethod
    def shuffle() -> CrossoverFunc:
        
        def func(x: array1D, y: array1D):
            
            ofs1, ofs2 = get_copies(x, y)
            
            index = np.random.choice(np.arange(0, x.size), x.size, replace = False)
            
            ran = np.random.randint(0, x.size)
            
            for i in range(ran):
                ind = index[i]
                ofs1[ind] = y[ind]
                ofs2[ind] = x[ind]
            
            return ofs1, ofs2
            
        return func
    
    @staticmethod
    def uniform_window(window: int = 7) -> CrossoverFunc:

        base_uniform = Crossover.uniform()

        def func(x: np.ndarray, y: np.ndarray):

            if x.size % window != 0:
                raise ValueError(f"dimension {x.size} cannot be divided by window {window}")
            
            items = int(x.size/window)

            zip_x, zip_y = base_uniform(np.zeros(items), np.ones(items))
            
            ofs1 = np.empty(x.size)
            ofs2 = np.empty(x.size)
            for i in range(items):
                sls = slice(i*window, (i+1)*window, 1)
                if zip_x[i] == 0:
                    ofs1[sls] = x[sls]
                    ofs2[sls] = y[sls]
                else:
                    ofs2[sls] = x[sls]
                    ofs1[sls] = y[sls]                    

            return ofs1, ofs2

        return func

    #
    #
    # ONLY FOR REAL VARIABLES
    #
    #
    
    @staticmethod
    def arithmetic() -> CrossoverFunc:
        
        def func(x: array1D, y: array1D):
            b = random.random()
            a = 1-b
            return a*x + b*y, a*y + b*x
        
        return func
    
    @staticmethod
    def mixed(alpha: float = 0.5) -> CrossoverFunc:
        
        def func(x: array1D, y: array1D):
            
            a = np.empty(x.size)
            b = np.empty(y.size)
            
            x_min = np.minimum(x, y)
            x_max = np.maximum(x, y)
            delta = alpha*(x_max-x_min)
            
            for i in range(x.size):
                a[i] = np.random.uniform(x_min[i] - delta[i], x_max[i] + delta[i])
                b[i] = np.random.uniform(x_min[i] + delta[i], x_max[i] - delta[i])
            
            return a, b
        
        return func
        


