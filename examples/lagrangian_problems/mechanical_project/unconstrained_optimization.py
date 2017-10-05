'''
Compares derivatives from the three implemented methods:

1- Finite difference scheme (default_grad_f_approx)
2- Direct method
3- Adjoint method
'''

import functools

from examples.lagrangian_problems.mechanical_project.constants import h_ini
from examples.lagrangian_problems.mechanical_project.linear_system import f, dfdx_adjoint, \
    dfdx_direct
from optimization import armijo_backtracking_line_search, default_stop_criterea, conjugate_gradient
from optimization import default_grad_f_approx
from optimization.defaults import DEFAULT_STRATEGY_FUNCTIONS
from optimization.minimize import minimize
import numpy as np

def stop_criterea_with_print_x_path(f, x, it, strategy_functions_dict):
    print("x=%s" % (x,))
    return default_stop_criterea(f, x, it, strategy_functions_dict, tol=1e-2, max_iter=10)


for grad_f in [
    default_grad_f_approx,
    lambda f, x: dfdx_direct(x),
    lambda f, x: dfdx_adjoint(x),
]:
    strategy_functions_dict = DEFAULT_STRATEGY_FUNCTIONS
    strategy_functions_dict['compute_step'] = functools.partial(armijo_backtracking_line_search, alpha_i=10.0)
    strategy_functions_dict['stop_criterea'] = stop_criterea_with_print_x_path
    strategy_functions_dict['compute_direction'] = conjugate_gradient()
    strategy_functions_dict['grad_f'] = grad_f
    initial_guess = np.array([h_ini, h_ini, h_ini])
    x_star = minimize(f, initial_guess, strategy_functions_dict)

    print("x_min = %s" % (x_star,))
    print("f(x_min) = %s" % (f(x_star),))
    print("")
