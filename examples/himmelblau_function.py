'''
Tries to minimize the Himmelblau's function given by

f(x) = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
It has one local maximum at f(-0.270845, -0.923039) = 181.617 and four local minima:
f(3.0, 2.0) = 0.0
f(-2.805118, 3.131312) = 0.0
f(-3.77931, -3.283186) = 0.0
f(3.584428, -1.848126) = 0.0
'''
import logging
import sys

from optimization.defaults import DEFAULT_STRATEGY_FUNCTIONS
from optimization.minimize import minimize
from optimization.plotting import FunctionPlotterHelper
import matplotlib.pyplot as plt
import numpy as np


ENABLE_PLOT = True
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

f = lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
initial_guess = np.array([0.1, 0.1])

strategy_functions_dict = DEFAULT_STRATEGY_FUNCTIONS

if ENABLE_PLOT:
    p = FunctionPlotterHelper(f)
    p.set_x_range(-6.0, 6.0)
    p.set_y_range(-6.0, 6.0)
    levels = np.r_[
        np.linspace(0.0, 50.0, 5, endpoint=False),
        np.linspace(50.0, 180.0, 6, endpoint=False),
    ]
    p.set_levels(levels)
    strategy_functions_dict['plot_helper'] = p.save_data_for_plot
    p.draw_function()

x_star = minimize(f, initial_guess, strategy_functions_dict)
print("x_min = %s" % (x_star,))
print("f(x_min) = %s" % (f(x_star),))

if ENABLE_PLOT:
    print("Building plot (May take a while)...")
    p.draw_solution_path()
    plt.show()
