'''
Tries to minimize the Sphere function given by

f(x) = sum{i = 0..n}(x[i] ** 2)

The function has been extended so that each axis can have a different starting point:

f(x) = sum{i = 0..n}((x[i] - p0[i]) ** 2)
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

sphere_f = lambda n, p0, x: sum([(x[i] - p0[i]) ** 2 for i in range(n)])
N_DIM = 2
f = lambda x: sphere_f(N_DIM, np.arange(0, N_DIM), x)

strategy_functions_dict = DEFAULT_STRATEGY_FUNCTIONS
strategy_functions_dict['stop_criterea'] = functools.partial(default_stop_criterea, tol=1e-2)
strategy_functions_dict['compute_step'] = functools.partial(armijo_backtracking_line_search, alpha_i=20.0)

if ENABLE_PLOT:
    p = FunctionPlotterHelper(f)
    p.set_x_range(-10.0, 10.0)
    p.set_y_range(-10.0, 10.0)
    levels = np.r_[
        np.linspace(0.0, 20.0, 4, endpoint=False),
        np.linspace(20.0, 100.0, 6, endpoint=False),
        np.linspace(100.0, 120.0, 3, endpoint=False),
    ]
    p.set_levels(levels)

    strategy_functions_dict['plot_helper'] = p.save_data_for_plot
    p.draw_function()
initial_guess = np.array([10.0] * N_DIM)

x_star = minimize(f, initial_guess)

print("x_min = %s" % (x_star,))
print("f(x_min) = %s" % (f(x_star),))

if ENABLE_PLOT:
    p.draw_solution_path()
    import matplotlib.pyplot as plt
    plt.show()
