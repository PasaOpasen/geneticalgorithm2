
from typing import List, Optional, Callable, Tuple, Sequence

import os
import random

import numpy as np

from OppOpPopInit import OppositionOperators, SampleInitializers

from .aliases import TypeAlias, array1D, array2D, PathLike
from .files import mkdir_of_file, mkdir

from .classes import MiddleCallbackData, Generation

from .utils import union_to_matrix, fast_max

from .crossovers import CrossoverFunc
from .selections import SelectionFunc
from .mutations import MutationFunc


CallbackFunc: TypeAlias = Callable[[int, List[float], array2D, array1D], None]
MiddleCallbackActionFunc: TypeAlias = Callable[[MiddleCallbackData], MiddleCallbackData]
MiddleCallbackConditionFunc: TypeAlias = Callable[[MiddleCallbackData], bool]
MiddleCallbackFunc: TypeAlias = Callable[[MiddleCallbackData], Tuple[MiddleCallbackData, bool]]


class Callbacks:
    
    @staticmethod
    def NoneCallback():
        return lambda generation_number, report_list, last_population, last_scores: None

    @staticmethod
    def SavePopulation(folder: PathLike, save_gen_step: int = 50, file_prefix: str = 'population') -> CallbackFunc:
        
        mkdir(folder)

        def func(generation_number: int, report_list: List[float], last_population: array2D, last_scores: array1D):
    
            if generation_number % save_gen_step != 0:
                return

            Generation(last_population, last_scores).save(
                os.path.join(
                    folder,
                    f"{file_prefix}_{generation_number}.npz"
                )
            )
        
        return func
    
    @staticmethod
    def PlotOptimizationProcess(
        folder: PathLike,
        save_gen_step: int = 50,
        show: bool = False,
        main_color: str = 'green',
        file_prefix: str = 'report'
    ) -> CallbackFunc:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        
        mkdir(folder)

        def func(generation_number: int, report_list: List[float], last_population: array2D, last_scores: array1D):

            if generation_number % save_gen_step != 0:
                return

            # if len(report_list) == 0:
            #     sys.stdout.write("No results to plot!\n")
            #     return
            
            ax = plt.axes()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            plt.plot(
                np.arange(1, 1 + len(report_list)),
                report_list,
                color=main_color,
                label='best of generation',
                linewidth=2
            )
            
            plt.xlabel('Generation')
            plt.ylabel('Minimized function')
            plt.title('GA optimization process')
            plt.legend()
            
            plt.savefig(os.path.join(folder, f"{file_prefix}_{generation_number}.png"), dpi=200)

            if show:
                plt.show()
            else:
                plt.close()
        
        return func


class Actions:

    @staticmethod
    def Stop(reason_name: str = 'stopped by Stop callback') -> MiddleCallbackActionFunc:
        
        def func(data: MiddleCallbackData):
            data.reason_to_stop = reason_name
            return data
        return func

    @staticmethod
    def ReduceMutationProb(reduce_coef: float = 0.9) -> MiddleCallbackActionFunc:
        
        def func(data: MiddleCallbackData):
            data.mutation_prob *= reduce_coef
            return data
        
        return func


    #def DualStrategyStep():
    #    pass

    #def SetFunction():
    #    pass


    @staticmethod
    def ChangeRandomCrossover(
        available_crossovers: Sequence[CrossoverFunc]
    ) -> MiddleCallbackActionFunc:

        def func(data: MiddleCallbackData):
            data.crossover = random.choice(available_crossovers)
            return data

        return func
    
    @staticmethod
    def ChangeRandomSelection(available_selections: Sequence[SelectionFunc]) -> MiddleCallbackActionFunc:

        def func(data: MiddleCallbackData):
            data.selection = random.choice(available_selections)
            return data

        return func

    @staticmethod
    def ChangeRandomMutation(available_mutations: Sequence[MutationFunc]) -> MiddleCallbackActionFunc:

        def func(data):
            data.mutation = random.choice(available_mutations)
            return data

        return func
    

    @staticmethod
    def RemoveDuplicates(
        oppositor: Optional[Callable[[array1D], array1D]] = None,
        creator: Optional[Callable[[], array1D]] = None,
        converter: Optional[Callable[[array1D], array1D]] = None
    ) -> MiddleCallbackActionFunc:
        """
        Removes duplicates from population

        Parameters
        ----------
        oppositor : oppositor from OppOpPopInit, optional
            oppositor for applying after duplicates removing. By default -- using just random initializer from creator.
                The default is None.
        creator : the function creates population samples, optional
            the function creates population samples if oppositor is None. The default is None.
        converter : func, optional
            function converts population samples in new format to compare (if needed). The default is None.

        """

        if creator is None and oppositor is None:
            raise Exception("No functions to fill population! creator or oppositors must be not None")

        if converter is None:
            def without_dup(pop: array2D, scores: array1D) -> Tuple[array2D, int]:
                """returns population without dups"""
                _, index_of_dups = np.unique(pop, axis=0, return_index=True)
                return union_to_matrix(pop[index_of_dups], scores[index_of_dups]), pop.shape[0] - index_of_dups.size
        else:
             def without_dup(pop: array2D, scores: array1D) -> Tuple[array2D, int]:
                """returns population without dups"""
                _, index_of_dups = np.unique(
                    np.array([converter(pop[i]) for i in range(pop.shape[0])]), axis=0, return_index=True
                )
                return union_to_matrix(pop[index_of_dups], scores[index_of_dups]), pop.shape[0] - index_of_dups.size


        if oppositor is None:
            def remover(pop: array2D, scores: array1D, set_function: Callable[[array2D], array1D]) -> array2D:

                pp, count_to_create = without_dup(pop, scores)  # pop without dups
                pp2 = np.empty((count_to_create, pp.shape[1])) 
                pp2[:, :-1] = SampleInitializers.CreateSamples(creator, count_to_create)  # new pop elements
                pp2[:, -1] = set_function(pp2[:, :-1])  # new elements values

                new_pop = np.vstack((pp, pp2))

                return new_pop[np.argsort(new_pop[:, -1])]  # new pop

        else:  # using oppositors
            def remover(pop: array2D, scores: array1D, set_function: Callable[[array2D], array1D]) -> array2D:

                pp, count_to_create = without_dup(pop, scores)  # pop without dups

                if count_to_create > pp.shape[0]:
                    raise Exception("Too many duplicates, cannot oppose")
                
                if count_to_create == 0:
                    return pp[np.argsort(pp[:, -1])]
                
                pp2 = np.empty((count_to_create, pp.shape[1])) 
                # oppose count_to_create worse elements
                pp2[:, :-1] = OppositionOperators.Reflect(pp[-count_to_create:, :-1], oppositor)  # new pop elements
                pp2[:, -1] = set_function(pp2[:, :-1])  # new elements values
                    
                new_pop = np.vstack((pp, pp2))
                    
                return new_pop[np.argsort(new_pop[:, -1])]  # new pop

        def func(data: MiddleCallbackData):
            new_pop = remover(
                data.last_generation.variables,
                data.last_generation.scores,
                data.set_function
            )

            data.last_generation = Generation.from_pop_matrix(new_pop)

            return data

        
        return func
    
    @staticmethod
    def CopyBest(by_indexes: Sequence[int]) -> MiddleCallbackActionFunc:
        """
        Copies best population object values (from dimensions in by_indexes) to all population
        """

        if type(by_indexes) != np.ndarray:
            by_indexes = np.array(by_indexes)

        def func(data: MiddleCallbackData):

            pop = data.last_generation.variables
            scores = data.last_generation.scores

            pop[:, by_indexes] = pop[np.argmin(scores), by_indexes]

            data.last_generation = Generation(pop, data.set_function(pop))

            return data
        return func

    @staticmethod
    def PlotPopulationScores(
        title_pattern: Callable[[MiddleCallbackData], str] = lambda data: f"Generation {data.current_generation}",
        save_as_name_pattern: Optional[Callable[[MiddleCallbackData], str]] = None
    ) -> MiddleCallbackActionFunc:
        """
        plots population scores
        needs 2 functions like data->str for title and file name
        """
        from .plotting_tools import plot_pop_scores

        use_save_as = (lambda data: None) if save_as_name_pattern is None else save_as_name_pattern

        def local_plot_callback(data: MiddleCallbackData):

            plot_pop_scores(
                data.last_generation.scores,
                title=title_pattern(data),
                save_as=use_save_as(data)
            )

            return data

        return local_plot_callback


class ActionConditions:
    
    @staticmethod
    def EachGen(generation_step: int = 10) -> MiddleCallbackConditionFunc:

        if generation_step < 1 or type(generation_step) is not int:
            raise Exception(f"Invalid generation step {generation_step}! Should be int and >=1")
        
        if generation_step == 1:
            return ActionConditions.Always()

        def func(data: MiddleCallbackData):
            return data.current_generation % generation_step == 0 and data.current_generation > 0

        return func

    @staticmethod
    def Always() -> MiddleCallbackConditionFunc:
        """
        makes action each generation
        """
        return lambda data: True

    @staticmethod
    def AfterStagnation(stagnation_generations: int = 50) -> MiddleCallbackConditionFunc:

        def func(data: MiddleCallbackData):
            return data.current_stagnation % stagnation_generations == 0 and data.current_stagnation > 0
        return func


    @staticmethod
    def Several(conditions: Sequence[MiddleCallbackConditionFunc]) -> MiddleCallbackConditionFunc:
        """
        returns function which checks all conditions from conditions
        """

        def func(data: MiddleCallbackData):
            return all(cond(data) for cond in conditions)
        
        return func


class MiddleCallbacks:

    @staticmethod
    def UniversalCallback(
        action: MiddleCallbackActionFunc,
        condition: MiddleCallbackConditionFunc,
        set_data_after_callback: bool = True
    ) -> MiddleCallbackFunc:
        
        def func(data: MiddleCallbackData):

            cond = condition(data)
            if cond:
                data = action(data)

            return data, (cond if set_data_after_callback else False)
        
        return func

    @staticmethod
    def ReduceMutationGen(
        reduce_coef: float = 0.9,
        min_mutation: float = 0.005,
        reduce_each_generation: int = 50,
        reload_each_generation: int = 500
    ) -> MiddleCallbackFunc:

        start_mutation = None

        def func(data: MiddleCallbackData):
            nonlocal start_mutation
            
            gen = data.current_generation
            mut = data.mutation_prob

            if start_mutation is None:
                start_mutation = mut

            c1 = gen % reduce_each_generation == 0
            c2 = gen % reload_each_generation == 0

            if c2:
                mut = start_mutation
            elif c1:
                mut *= reduce_coef
                mut = fast_max(mut, min_mutation)

            data.mutation_prob = mut

            return data, (c1 or c2)

        return func

    #def ReduceMutationStagnation(reduce = 0.5, stagnation_gens = 50):
    #    pass 

    @staticmethod
    def GeneDiversityStats(step_generations_for_plotting: int = 10) -> MiddleCallbackFunc:

        if step_generations_for_plotting < 1:
            raise Exception(f"Wrong step = {step_generations_for_plotting}, should be int and > 0!!")

        import matplotlib.pyplot as plt

        div = []
        count = []
        most = []


        def func(data: MiddleCallbackData):

            nonlocal div, count, most

            dt = data.last_generation.variables

            uniq, counts = np.unique(dt, return_counts=True, axis=0)
            # raise Exception()
            count.append(counts.size)
            most.append(counts.max())

            gene_diversity = 0
            for index in range(dt.shape[0]-1):
                gene_diversity += np.count_nonzero(dt[index, :] != dt[index:, :]) / (dt.shape[0] - index)
            div.append(gene_diversity/dt.shape[1])

            if data.current_generation % step_generations_for_plotting == 0:

                fig, axes = plt.subplots(3, 1)

                (ax1, ax2, ax3) = axes

                ax1.plot(count)
                #axs[0, 0].set_title('Axis [0, 0]')

                ax2.plot(most, 'tab:orange')
                #axs[0, 1].set_title('Axis [0, 1]')

                ax3.plot(div, 'tab:green')
                #axs[1, 0].set_title('Axis [1, 0]')

                ylabs = [
                    'Count of unique objects',
                    'Count of most popular object',
                    'Simple gene diversity'
                ]

                for i, ax in enumerate(axes):
                    ax.set(xlabel='Generation number')
                    ax.set_title(ylabs[i])

                # Hide x labels and tick labels for top plots and y ticks for right plots.
                for ax in axes:
                    ax.label_outer()

                fig.suptitle(f'Diversity report (pop size = {dt.shape[0]})')    
                fig.tight_layout()
                plt.show()

            return data, False

        return func






