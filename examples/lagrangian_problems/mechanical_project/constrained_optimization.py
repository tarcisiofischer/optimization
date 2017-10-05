import functools
import logging
import sys

from examples.lagrangian_problems.mechanical_project.constants import V_max, \
    h_ini
from examples.lagrangian_problems.mechanical_project.constraints import dg1_dx, g1, dg2_dx, g2, \
    dg3_dx, g3, dg4_dx, g4, dg5_dx, g5, dg6_dx, g6, dg7_dx, g7, V
from examples.lagrangian_problems.mechanical_project.linear_system import f, dfdx_adjoint
from optimization import armijo_backtracking_line_search, conjugate_gradient, residual_based_stop_criterea
from optimization.defaults import DEFAULT_STRATEGY_FUNCTIONS
from optimization.minimize import minimize
from optimization.utils import build_augmented_lagrangian
import numpy as np

'''
Solves the Augmented Lagrangian constrained optimization problem.
'''

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class stop_criterea_printing_results(residual_based_stop_criterea):
    def __call__(self, f, x, it, strategy_functions_dict):
        gf = strategy_functions_dict['grad_f'](f, x)
        print("Iteration %s" % (it,))
        print("x=(%.3f, %.3f, %.3f)" % (x[0], x[1], x[2],))
        print("mi=%.3f" % (x[9],))
        print("f(x)=%s" % (f(x),))
        print("gf=%s" % (gf,))
        print("%s & (%.3f, %.3f, %.3f) & %.3f & %s & %s" % (it, x[0], x[1], x[2], x[9], f(x), gf[0:3]))
        print("")
        return residual_based_stop_criterea.__call__(self, f, x, it, strategy_functions_dict, tol=1e-2, max_iter=50)

initial_guess = np.array([h_ini, h_ini, h_ini])
rho_volume = 1000.0
for i in range(4):
    print("rho_volume=%s" % (rho_volume,))

    inequality_constraints = [
        (g1, dg1_dx, 10.0),  # h1 <= hmax
        (g2, dg2_dx, 10.0),  # h2 <= hmax
        (g3, dg3_dx, 10.0),  # h3 <= hmax
        (g4, dg4_dx, 10.0),  # h1 >= hmin
        (g5, dg5_dx, 10.0),  # h2 >= hmin
        (g6, dg6_dx, 10.0),  # h3 >= hmin
        (g7, dg7_dx, rho_volume),  # h1*b*L + h2*b*L + h3*b*L <= Vmax
    ]
    augmented_L, grad_augmented_L = build_augmented_lagrangian(
        f,
        3,
        [],
        [c[0] for c in inequality_constraints],
        [c[2] for c in inequality_constraints],
        dfdx_adjoint,
        [],
        [c[1] for c in inequality_constraints],
    )
    strategy_functions_dict = DEFAULT_STRATEGY_FUNCTIONS
    strategy_functions_dict['compute_step'] = functools.partial(armijo_backtracking_line_search, alpha_i=1.0, beta=0.1)
    strategy_functions_dict['stop_criterea'] = stop_criterea_printing_results()
    strategy_functions_dict['grad_f'] = lambda f, x: grad_augmented_L(x)
    strategy_functions_dict['compute_direction'] = conjugate_gradient()

    x_star = minimize(augmented_L, np.r_[initial_guess, [0.0] * len(inequality_constraints)], strategy_functions_dict)

    print("x* = %s" % (x_star[0:3],))
    print("f(x*) = %s" % (f(x_star),))
    print("volume = %s / %s" % (V(x_star), V_max,))
    print("mi = %s" % (x_star[3:],))
    print("")
    initial_guess = x_star[0:3]
    rho_volume *= 10.0

print("=" * 10)
print("x_min = %s" % (x_star,))
print("f(x_min) = %s" % (f(x_star),))
print("volume = %s / %s" % (V(x_star), V_max,))
