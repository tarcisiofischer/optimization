'''
Minimize f(x,y) = x**2 - y**2
Subject to x**2 + y**2 <= 1
'''
import functools
import logging
import sys

from optimization import default_stop_criterea, armijo_backtracking_line_search
from optimization.defaults import DEFAULT_STRATEGY_FUNCTIONS
from optimization.minimize import minimize
from optimization.plotting import FunctionPlotterHelper
from optimization.utils import build_augmented_lagrangian
import matplotlib.pyplot as plt
import numpy as np


ENABLE_PLOT = True
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

f = lambda x: x[0] ** 2 - x[1] ** 2
g_ = lambda x: x[0] ** 2 + x[1] ** 2
g = lambda x: g_(x) - 1

grad_f = lambda x: np.array([
    2 * x[0],  # df/dx
    - 2 * x[1],  # df/dy
])
grad_g = lambda x: np.array([
    2 * x[0],  # dg/dx
    2 * x[1],  # dg/dy
])

phi = 10000.0
L, grad_L = build_augmented_lagrangian(
    f,
    2,
    [],
    [g],
    phi,
    grad_f,
    [],
    [grad_g],
)

initial_guesses = [
    (np.r_[np.array([1e-8, 1e-8]), [0.0]] , 'g:'),
    (np.r_[np.array([0.1, 0.5]), [0.0]] , 'r:'),
    (np.r_[np.array([0.8, -0.5]), [0.0]] , 'b:'),
    (np.r_[np.array([-0.9, 0.1]), [0.0]] , 'y:'),
]

strategy_functions_dict = DEFAULT_STRATEGY_FUNCTIONS
strategy_functions_dict['stop_criterea'] = functools.partial(default_stop_criterea, tol=1e-6, max_iter=800)
strategy_functions_dict['compute_step'] = functools.partial(armijo_backtracking_line_search, alpha_i=1.0)
strategy_functions_dict['grad_f'] = lambda f, x: grad_L(x)
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

for initial_guess, solution_plot_style in initial_guesses:
    x_star = minimize(L, initial_guess, strategy_functions_dict)
    print("x_min = %s" % (x_star,))
    print("f(x_min) = %s" % (f(x_star),))

    if ENABLE_PLOT:
        p.draw_solution_path(solution_plot_style)

if ENABLE_PLOT:
    # Show the constrain level curve
    X, Y = np.meshgrid(p._xrange, p._yrange)
    plt.contour(X, Y, g_(np.array([X, Y])), [+1.0])

plt.show()
