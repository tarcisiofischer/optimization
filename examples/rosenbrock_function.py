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
import numpy as np


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

rosenbrock = lambda a, b, x: (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
f = lambda x: rosenbrock(1.0, 100.0, x)

strategy_functions_dict = DEFAULT_STRATEGY_FUNCTIONS
strategy_functions_dict['stop_criterea'] = functools.partial(default_stop_criterea, tol=1e-2, max_iter=2000)
# strategy_functions_dict['compute_step'] = functools.partial(equal_interval_search, tol=1e-4)
strategy_functions_dict['compute_step'] = functools.partial(armijo_backtracking_line_search, alpha_i=2.0)

x_star = minimize(f, np.array([-3.0, -4.0]), strategy_functions_dict)
print(x_star)
print(f(x_star))

# print(f(np.array([1, 1])))
