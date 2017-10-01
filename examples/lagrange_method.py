'''
Minimize f(x,y) = x + y
Subject to x**2 + y**2 = 1

The equivalent Lagrangian problem is given by

f(x, y) = x + y
h(x, y) = x**2 + y**2 - 1
L(x, y, l) = f(x,y) + l * h(x,y)
'''
import functools
import logging
import sys

from optimization import default_stop_criterea, armijo_backtracking_line_search
from optimization.defaults import DEFAULT_STRATEGY_FUNCTIONS
from optimization.minimize import minimize
from optimization.plotting import FunctionPlotterHelper
import matplotlib.pyplot as plt
import numpy as np


ENABLE_PLOT = True
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

f = lambda x: x[0] + x[1]
h_ = lambda x: x[0] ** 2 + x[1] ** 2
h = lambda x: h_(x) - 1
L = lambda x: f(x) + x[2] * h(x)
grad_L = lambda f, x: np.array([
    1 + 2 * x[2] * x[0],
    1 + 2 * x[2] * x[1],
    h(x)
])

initial_guesses = [
    # Good initial guess
    np.array([0.0, 0.0, 0.0]),

    # Bad initial guesses
    np.array([0.0, 0.0, 1.0]),  # Lambda != 0.0
    np.array([0.0, 1.0, 0.0]),  # Different starting point
    np.array([0.0, -1.0, 0.0]),  # Different starting point
]
solution_styles = [
    'g:',
    'r:',
    'r:',
    'r:',
]

strategy_functions_dict = DEFAULT_STRATEGY_FUNCTIONS
strategy_functions_dict['stop_criterea'] = functools.partial(default_stop_criterea, tol=1e-6, max_iter=1000)
strategy_functions_dict['compute_step'] = functools.partial(armijo_backtracking_line_search, alpha_i=0.1)
strategy_functions_dict['grad_f'] = grad_L
if ENABLE_PLOT:
    p = FunctionPlotterHelper(f)
    p.set_x_range(-2.0, 2.0)
    p.set_y_range(-2.0, 2.0)
    levels = np.r_[
        np.linspace(-4.0, 4.0, 10, endpoint=False),
    ]
    p.set_levels(levels)
    strategy_functions_dict['plot_helper'] = p.save_data_for_plot
    p.draw_function()

for initial_guess, solution_plot_style in zip(initial_guesses, solution_styles):
    x_star = minimize(L, initial_guess, strategy_functions_dict)
    print("x_min = %s" % (x_star,))
    print("f(x_min) = %s" % (f(x_star),))

    if ENABLE_PLOT:
        p.draw_solution_path(solution_plot_style)

if ENABLE_PLOT:
    # Show the constrain level curve
    X, Y = np.meshgrid(p._xrange, p._yrange)
    plt.contour(X, Y, h_(np.array([X, Y])), [+1.0])

plt.show()
