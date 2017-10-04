from mpl_toolkits.mplot3d.axes3d import Axes3D

from examples.lagrangian_problems.mechanical_project.constants import h_max, h_min, V_max
from examples.lagrangian_problems.mechanical_project.constraints import V
from examples.lagrangian_problems.mechanical_project.linear_system import f
from optimization.plotting import FunctionPlotterHelper
import matplotlib.pyplot as plt
import numpy as np


h3 = 40.0

# 3D function visualization
xrange = np.arange(10.0, 80.0, 1.e-1)
yrange = np.arange(10.0, 80.0, 1.e-1)
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

plt.show()
