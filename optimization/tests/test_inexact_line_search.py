import pytest

from optimization import armijo_backtracking_line_search
from optimization.defaults import DEFAULT_STRATEGY_FUNCTIONS
import numpy as np


def test_armijo_line_search():
    def direction_provider(x):
        '''
        Give a search direction for the function f(x) = x**2
        '''
        if x > 0.0:
            return np.array([-1.0])
        else:
            return np.array([+1.0])

    f = lambda x: x[0] ** 2

    alpha = np.array([-11.0])

    for _ in range(10):
        x0 = alpha
        d = direction_provider(x0)
        alpha = armijo_backtracking_line_search(
            f,
            d,
            x0,
            DEFAULT_STRATEGY_FUNCTIONS,

            # Purposeful exaggerated step will generate cases where the search will pass the
            # minimum, and the algorithm will be forced to go back.
            alpha_i=5.0,
        )
        assert f(alpha) < f(x0)
