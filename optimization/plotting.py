import pylab

import matplotlib.pyplot as plt
import numpy as np


class FunctionPlotterHelper(object):

    def __init__(self, f):
        self.DEFAULT_CMAP = plt.cm.BuPu
        self.data_for_plot = []
        self._f = f
        self.set_x_range(-10.0, 10.0)
        self.set_y_range(-10.0, 10.0)
        self._levels = None
        plt.figure()

    def save_data_for_plot(self, x, it):
        self.data_for_plot.append(np.array(x))

    def set_x_range(self, min_, max_, delta=0.1):
        self._xrange = np.arange(min_, max_, delta)

    def set_y_range(self, min_, max_, delta=0.1):
        self._yrange = np.arange(min_, max_, delta)

    def set_levels(self, levels):
        self._levels = levels


    def draw_function(self, broadcastable=True):
        X, Y = np.meshgrid(self._xrange, self._yrange)

        if broadcastable:
            F = self._f(np.array([X, Y]))
        else:
            F = np.zeros_like(X)
            for ii, xx in enumerate(self._xrange):
                for jj, yy in enumerate(self._yrange):
                    F[ii][jj] = self._f(np.array([xx, yy]))

        if self._levels is not None:
            cp = plt.contour(X, Y, F, self._levels, cmap=self.DEFAULT_CMAP)
        else:
            cp = plt.contour(X, Y, F, cmap=self.DEFAULT_CMAP)
        plt.colorbar(cp)
        plt.clabel(cp, inline=1, fontsize=5)

        pylab.xlim(self._xrange[0], self._xrange[-1])
        pylab.ylim(self._yrange[0], self._yrange[-1])


    def draw_solution_path(self, style='g:'):
        # Plot the solution path
        data_for_plot = np.array(self.data_for_plot)
        plt.plot(data_for_plot[:, 0], data_for_plot[:, 1], style)

        # Plot the final solution point
        ax = plt.gca()
        ax.scatter(self.data_for_plot[-1][0], self.data_for_plot[-1][1], s=8, c='r')

        self.data_for_plot = []
