
from typing import List

import os

import numpy as np

from ..data_types.generation import Generation
from .data import SimpleCallbackFunc

from ..utils.aliases import PathLike, array2D, array1D
from ..utils.files import mkdir


class Callbacks:
    """
    Static class with several simple callback methods
    """

    @staticmethod
    def NoneCallback():
        return lambda generation_number, report_list, last_population, last_scores: None

    @staticmethod
    def SavePopulation(
        folder: PathLike, save_gen_step: int = 50, file_prefix: str = 'population'
    ) -> SimpleCallbackFunc:
        """saves population to disk periodically"""

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
    ) -> SimpleCallbackFunc:
        """
        Saves optimization process plots to disk periodically
        Args:
            folder:
            save_gen_step:
            show:
            main_color:
            file_prefix:

        Returns:

        """
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
