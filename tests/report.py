
import numpy as np

from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import plot_several_lines

def f(X):
    return 50*np.sum(X) - np.sum(np.sqrt(X) * np.sin(X))

dim = 25
varbound = [[0 ,10]]*dim

model = ga(function=f, dimension=dim,
           variable_type='real', variable_boundaries=varbound,
           algorithm_parameters={
               'max_num_iteration': 600
           }
)

# here model exists and has checked_reports field
# now u can append any functions to report

model.checked_reports.extend(
    [
        ('report_average', np.mean),
        ('report_25', lambda arr: np.quantile(arr, 0.25)),
        ('report_50', np.median)
    ]
)

# run optimization process
model.run(no_plot = False)

# now u have not only model.report but model.report_25 and so on

#plot reports
names = [name for name, _ in model.checked_reports[::-1]]
plot_several_lines(
    lines=[getattr(model, name) for name in names],
    colors=('green', 'black', 'red', 'blue'),
    labels=['median value', '25% quantile', 'mean of population', 'best pop score'],
    linewidths=(1, 1.5, 1, 2),
    title="Several custom reports with base reports",
    save_as='./output/report.png'
)


