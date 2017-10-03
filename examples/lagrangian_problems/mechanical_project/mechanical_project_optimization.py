import functools
import logging
import sys

from examples.lagrangian_problems.mechanical_project.constants import P_y_4, h_min, h_max, b, V_max, \
    h_ini, L
from examples.lagrangian_problems.mechanical_project.linear_system import u, dudx
from optimization import armijo_backtracking_line_search, default_stop_criterea
from optimization.defaults import DEFAULT_STRATEGY_FUNCTIONS
from optimization.minimize import minimize
from optimization.utils import build_augmented_lagrangian
import numpy as np


# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


f = lambda x: 0.5 * P_y_4 * u(x)[4]
pdf_dx = lambda x: 0.0
pdf_du = lambda x: 0.5 * P_y_4
pdu_dx = lambda x: dudx(x)[:, 4]
df_dx = lambda x: pdf_dx(x) + pdf_du(x) * pdu_dx(x)

g1 = lambda x: (x[0] - h_max) / h_max
dg1_dx = lambda x: np.array([
    1.0 / h_max,
    0.0,
    0.0,
])
g2 = lambda x: (x[1] - h_max) / h_max
dg2_dx = lambda x: np.array([
    0.0,
    1.0 / h_max,
    0.0,
])
g3 = lambda x: (x[2] - h_max) / h_max
dg3_dx = lambda x: np.array([
    0.0,
    0.0,
    1.0 / h_max,
])

g4 = lambda x:(-x[0] + h_min) / h_min
dg4_dx = lambda x: np.array([
    - 1.0 / h_min,
    0.0,
    0.0,
])
g5 = lambda x:(-x[1] + h_min) / h_min
dg5_dx = lambda x: np.array([
    0.0,
    - 1.0 / h_min,
    0.0,
])
g6 = lambda x:(-x[2] + h_min) / h_min
dg6_dx = lambda x: np.array([
    0.0,
    0.0,
    - 1.0 / h_min
])
V = lambda x: x[0] * b * L + x[1] * b * L + x[2] * b * L
g7 = lambda x: (V(x) - V_max) / V_max
dg7_dx = lambda x: np.array([
    b * L / V_max,
    b * L / V_max,
    b * L / V_max,
])

initial_guess = np.array([h_ini, h_ini, h_ini])
rho_volume = 1000.0
for i in range(5):
    print("rho_volume=%s" % (rho_volume,))
    inequality_constraints = [
        (g1, dg1_dx, 1.0),  # h1 <= hmax
        (g2, dg2_dx, 1.0),  # h2 <= hmax
        (g3, dg3_dx, 1.0),  # h3 <= hmax
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
        df_dx,
        [],
        [c[1] for c in inequality_constraints],
    )

    strategy_functions_dict = DEFAULT_STRATEGY_FUNCTIONS
    strategy_functions_dict['grad_f'] = lambda f, x: grad_augmented_L(x)
    strategy_functions_dict['compute_step'] = functools.partial(armijo_backtracking_line_search, alpha_i=0.2)
    strategy_functions_dict['stop_criterea'] = functools.partial(default_stop_criterea, tol=1e-2, max_iter=100)

    x_star = minimize(augmented_L, np.r_[initial_guess, [0.0] * len(inequality_constraints)], strategy_functions_dict)
    print(x_star[0:3])
    print("f(x_min) = %s" % (f(x_star),))
    print("volume = %s / %s" % (V(x_star), V_max,))
    print("")
    initial_guess = x_star[0:3]
    rho_volume *= 5.0

print("x_min = %s" % (x_star,))
print("f(x_min) = %s" % (f(x_star),))
print("volume = %s / %s" % (V(x_star), V_max,))
