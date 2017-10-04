import functools

from examples.lagrangian_problems.mechanical_project.constants import h_ini
from examples.lagrangian_problems.mechanical_project.linear_system import f, dfdx_adjoint,\
    dfdx_direct
from optimization import armijo_backtracking_line_search, default_stop_criterea
from optimization.defaults import DEFAULT_STRATEGY_FUNCTIONS
from optimization.minimize import minimize
import numpy as np
from optimization import default_grad_f_approx

'''
Compares derivatives from the three implemented methods:

1- Finite difference scheme (default_grad_f_approx)
2- Direct method
3- Adjoint method
'''

strategy_functions_dict = DEFAULT_STRATEGY_FUNCTIONS
strategy_functions_dict['compute_step'] = functools.partial(armijo_backtracking_line_search, alpha_i=10.0)

def stop_criterea_with_print_x_path(f, x, it, strategy_functions_dict):
    print("x=%s" % (x,))
    return default_stop_criterea(f, x, it, strategy_functions_dict, tol=1e-2, max_iter=10)

strategy_functions_dict['stop_criterea'] = stop_criterea_with_print_x_path

for grad_f in [
    default_grad_f_approx,
    lambda f, x: dfdx_direct(x)[:, 4],
    lambda f, x: dfdx_adjoint(x)[:, 4],
]:
    strategy_functions_dict['grad_f'] = grad_f
    initial_guess = np.array([h_ini, h_ini, h_ini])
    x_star = minimize(f, initial_guess, strategy_functions_dict)
    print("x_min = %s" % (x_star,))
    print("f(x_min) = %s" % (f(x_star),))
    print("")
