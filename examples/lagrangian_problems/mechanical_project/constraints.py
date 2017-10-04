from examples.lagrangian_problems.mechanical_project.constants import h_max, h_min, b, L, V_max
import numpy as np


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
