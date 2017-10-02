'''
Minimize the function

W(u(h)) = 1/2. * P_y_4 * u_y_4

where

Ku = F is a linear system that relates h with u (See implementation for details).
'''
from examples.lagrangian_problems.mechanical_project.constants import P_y_4, h_min, h_max, b, V_max, \
    h_ini, L
from examples.lagrangian_problems.mechanical_project.linear_system import u, du_dh1, du_dh2, du_dh3
from optimization.defaults import DEFAULT_STRATEGY_FUNCTIONS
from optimization.minimize import minimize
from optimization.utils import build_augmented_lagrangian
import numpy as np


f = lambda x: 0.5 * P_y_4 * u(x)[4]
df_du = lambda x: 0.5 * P_y_4
df_dx = lambda x: df_du(x) * np.array([
    0.0,
    0.0,
    0.0
])

g1 = lambda x: x[0] - h_min
dg1_dx = lambda x: np.array([
    1.0,
    0.0,
    0.0
])
g2 = lambda x: x[1] - h_min
dg2_dx = lambda x: np.array([
    0.0,
    1.0,
    0.0
])
g3 = lambda x: x[2] - h_min
dg3_dx = lambda x: np.array([
    0.0,
    0.0,
    1.0
])
g4 = lambda x:-x[0] + h_max
dg4_dx = lambda x: np.array([
    - 1.0,
    0.0,
    0.0
])
g5 = lambda x:-x[1] + h_max
dg5_dx = lambda x: np.array([
    0.0,
    - 1.0,
    0.0
])
g6 = lambda x:-x[2] + h_max
dg6_dx = lambda x: np.array([
    0.0,
    0.0,
    - 1.0
])
g7 = lambda x: x[0] * b * L + x[1] * b * L + x[2] * b * L - V_max
dg7_dx = lambda x: np.array([
    b * L,
    b * L,
    b * L,
])

augmented_L, grad_augmented_L = build_augmented_lagrangian(
    f,
    3,
    [],
    [g1, g2, g3, g4, g5, g6, g7],
    100.0,
    df_dx,
    [],
    [
        dg1_dx,
        dg2_dx,
        dg3_dx,
        dg4_dx,
        dg5_dx,
        dg6_dx,
        dg7_dx,
    ]
)

# augmented_L(np.array([40.0, 40.0, 40.0] + [0.0] * 7))
# grad_augmented_L(np.array([40.0, 40.0, 40.0] + [0.0] * 7))


strategy_functions_dict = DEFAULT_STRATEGY_FUNCTIONS
strategy_functions_dict['grad_f'] = lambda f, x: grad_augmented_L(x)

initial_guess = np.array([h_ini, h_ini, h_ini] + [0.0] * 7)
x_star = minimize(augmented_L, initial_guess, strategy_functions_dict)
print("x_min = %s" % (x_star,))
print("f(x_min) = %s" % (f(x_star),))
