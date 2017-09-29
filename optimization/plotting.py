import numpy as np
import pylab


class FunctionPlotterHelper(object):
    
    def __init__(self, f):
        self.data_for_plot = []
        self._f = f
        self.set_x_range(-10.0, 10.0)
        self.set_y_range(-10.0, 10.0)
        self._levels = None

    def save_data_for_plot(self, x, it):
        self.data_for_plot.append(np.array(x))

    def set_x_range(self, min_, max_, delta=0.1):
        self._xrange = np.arange(min_, max_, delta)

    def set_y_range(self, min_, max_, delta=0.1):
        self._yrange = np.arange(min_, max_, delta)

    def set_levels(self, levels):
        self._levels = levels

    def plot(self):
        import matplotlib.pyplot as plt
        X, Y = np.meshgrid(self._xrange, self._yrange)

        plt.figure()
        if self._levels:
            cp = plt.contour(X, Y, self._f(np.array([X, Y])), self._levels)
        else:
            cp = plt.contour(X, Y, self._f(np.array([X, Y])))

        plt.colorbar(cp)
        plt.clabel(cp, inline=1, fontsize=5)
        for point0, point1 in zip(self.data_for_plot[:-1], self.data_for_plot[1:]):
            plt.plot([point0[0], point1[0]], [point0[1], point1[1]], '--k')
        
        ax = plt.gca()
        ax.scatter(self.data_for_plot[-1][0], self.data_for_plot[-1][1], s=8, c='r')

        pylab.xlim(self._xrange[0], self._xrange[-1])
        pylab.ylim(self._yrange[0], self._yrange[-1])

        plt.show()
