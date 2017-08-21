from optimization._line_search import equal_interval_search
from optimization.defaults import DEFAULT_STRATEGY_FUNCTIONS
import numpy as np


def test_equally_spaced_line_search():
    f = lambda x: 2 - 4 * x[0] + np.e ** x[0]

    x0 = 0.0
    d = +1.0
    alpha = equal_interval_search(f, d, x0, DEFAULT_STRATEGY_FUNCTIONS)
    # Expected result computed on Arora's book - Seems not to be very close to the actual solution
    expected_alpha = 1.386511

    assert abs(alpha - expected_alpha) <= 1e-2
