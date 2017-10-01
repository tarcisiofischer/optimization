'''
Tries to minimize the Rosenbrock function given by

f(x) = (a - x[0])**2 + b*(x[1] * x[0]**2)**2
'''
import functools
import logging
import sys

from optimization import default_stop_criterea, armijo_backtracking_line_search
from optimization.defaults import DEFAULT_STRATEGY_FUNCTIONS
from optimization.minimize import minimize
from optimization.plotting import FunctionPlotterHelper
import numpy as np


ENABLE_PLOT = True
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

rosenbrock = lambda a, b, x: (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
f = lambda x: rosenbrock(1.0, 100.0, x)
initial_guess = np.array([-0.5, 2.5])

strategy_functions_dict = DEFAULT_STRATEGY_FUNCTIONS
strategy_functions_dict['stop_criterea'] = functools.partial(default_stop_criterea, tol=1e-3, max_iter=5000)
strategy_functions_dict['compute_step'] = functools.partial(armijo_backtracking_line_search, alpha_i=0.1)

if ENABLE_PLOT:
    p = FunctionPlotterHelper(f)
    p.set_x_range(-1.5, 1.5)
    p.set_y_range(-1.0, 3.0)
    levels = np.r_[
        np.linspace(0.0, 10.0, 3, endpoint=False),
        np.linspace(10.0, 30.0, 3, endpoint=False),
        np.linspace(30.0, 300.0, 6, endpoint=False),
    ]
    p.set_levels(levels)
    strategy_functions_dict['plot_helper'] = p.save_data_for_plot
    p.draw_function()

x_star = minimize(f, initial_guess, strategy_functions_dict)
print("x_min = %s" % (x_star,))
print("f(x_min) = %s" % (f(x_star),))

if ENABLE_PLOT:
    p.draw_solution_path()
    import matplotlib.pyplot as plt
    plt.show()
