import functools

from examples.lagrangian_problems.mechanical_project.constants import P_y_4, h_ini
from examples.lagrangian_problems.mechanical_project.linear_system import u, dudx, dfdx_adjoint
from optimization import armijo_backtracking_line_search, default_stop_criterea
from optimization.defaults import DEFAULT_STRATEGY_FUNCTIONS
from optimization.minimize import minimize
import numpy as np


f = lambda x: 0.5 * P_y_4 * u(x)[4]
# pdf_dx = lambda x: 0.0
# pdf_du = lambda x: 0.5 * P_y_4
# pdu_dx = lambda x: dudx(x)[:, 4]
# df_dx = lambda x: pdf_dx(x) + pdf_du(x) * pdu_dx(x)

df_dx = lambda x: dfdx_adjoint(x)[:, 4]

strategy_functions_dict = DEFAULT_STRATEGY_FUNCTIONS
strategy_functions_dict['grad_f'] = lambda f, x: df_dx(x)
strategy_functions_dict['compute_step'] = functools.partial(armijo_backtracking_line_search, alpha_i=10.0)
strategy_functions_dict['stop_criterea'] = functools.partial(default_stop_criterea, tol=1e-2, max_iter=600)

initial_guess = np.array([h_ini, h_ini, h_ini])
x_star = minimize(f, initial_guess, strategy_functions_dict)
print("x_min = %s" % (x_star,))
print("f(x_min) = %s" % (f(x_star),))
