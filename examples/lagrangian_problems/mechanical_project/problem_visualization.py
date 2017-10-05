from mpl_toolkits.mplot3d.axes3d import Axes3D

from examples.lagrangian_problems.mechanical_project.constants import h_max, h_min, V_max
from examples.lagrangian_problems.mechanical_project.constraints import V, dg7_dx, g7, dg6_dx, g6, \
    dg5_dx, g5, dg4_dx, g4, dg3_dx, g3, dg2_dx, g2, dg1_dx, g1
from examples.lagrangian_problems.mechanical_project.linear_system import f, dfdx_adjoint
from optimization.plotting import FunctionPlotterHelper
from optimization.utils import build_penalized_function
import matplotlib.pyplot as plt
import numpy as np


h3 = 40.0

# 3D function visualization
xrange = np.arange(10.0, 80.0, 1.e-0)
yrange = np.arange(10.0, 80.0, 1.e-0)
X, Y = np.meshgrid(xrange, yrange)
F = np.zeros_like(X)
for ii, xx in enumerate(xrange):
    for jj, yy in enumerate(yrange):
        F[ii][jj] = f(np.array([xx, yy, h3]))

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X, Y, F)
plt.draw()

# Contour plot visualization
f_2d = lambda x: f(np.r_[x, h3])
p = FunctionPlotterHelper(f_2d)
p.set_x_range(h_min, h_max, delta=1.0)
p.set_y_range(h_min, h_max, delta=1.0)
levels = np.r_[
    np.linspace(0.0, 30.0, 12, endpoint=False),
    np.linspace(30.0, 100.0, 8, endpoint=False),
    np.linspace(100.0, 1000.0, 8, endpoint=False),
]
p.set_levels(levels)
p.draw_function(broadcastable=False)

# Show the constrain level curve
X, Y = np.meshgrid(p._xrange, p._yrange)
plt.contour(X, Y, V(np.array([X, Y, np.ones_like(X) * h3])), [V_max])

# Penalized problem visualization
xrange = np.arange(10.0, 80.0, 1.e-0)
yrange = np.arange(10.0, 80.0, 1.e-0)
X, Y = np.meshgrid(xrange, yrange)
F = np.zeros_like(X)
inequality_constraints = [
    (g1, dg1_dx, 10.0),  # h1 <= hmax
    (g2, dg2_dx, 10.0),  # h2 <= hmax
    (g3, dg3_dx, 10.0),  # h3 <= hmax
    (g4, dg4_dx, 10.0),  # h1 >= hmin
    (g5, dg5_dx, 10.0),  # h2 >= hmin
    (g6, dg6_dx, 10.0),  # h3 >= hmin
    (g7, dg7_dx, 30000.0),  # h1*b*L + h2*b*L + h3*b*L <= Vmax
]
P, grad_P = build_penalized_function(
    f,
    3,
    [],
    [c[0] for c in inequality_constraints],
    [c[2] for c in inequality_constraints],
    dfdx_adjoint,
    [],
    [c[1] for c in inequality_constraints],
)
for ii, xx in enumerate(xrange):
    for jj, yy in enumerate(yrange):
        F[ii][jj] = P(np.array([xx, yy, h3]))

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X, Y, F)
plt.draw()

plt.show()
