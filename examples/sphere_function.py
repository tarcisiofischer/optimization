'''
Tries to minimize the Sphere function given by

f(x) = sum{i = 0..n}(x[i] ** 2)

The function has been extended so that each axis can have a different starting point:

f(x) = sum{i = 0..n}((x[i] - p0[i]) ** 2)
'''
import functools
import logging
import sys

from optimization import default_stop_criterea, equal_interval_search, \
    armijo_backtracking_line_search
from optimization.defaults import DEFAULT_STRATEGY_FUNCTIONS
from optimization.minimize import minimize
import numpy as np


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

sphere_f = lambda n, p0, x: sum([(x[i] - p0[i]) ** 2 for i in range(n)])

N_DIM = 5
f = lambda x: sphere_f(N_DIM, np.arange(0, N_DIM), x)

strategy_functions_dict = DEFAULT_STRATEGY_FUNCTIONS
strategy_functions_dict['stop_criterea'] = functools.partial(default_stop_criterea, tol=1e-2)
# strategy_functions_dict['compute_step'] = functools.partial(equal_interval_search, tol=1e-4)
strategy_functions_dict['compute_step'] = functools.partial(armijo_backtracking_line_search, alpha_i=2.0)

x_star = minimize(f, np.array([99.0] * N_DIM))

print(x_star)
print(f(x_star))
