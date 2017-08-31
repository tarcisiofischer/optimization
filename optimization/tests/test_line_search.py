from optimization._line_search import equal_interval_search
from optimization.defaults import DEFAULT_STRATEGY_FUNCTIONS
import numpy as np


_arora_example_f = lambda x: 2 - 4 * x[0] + np.e ** x[0]
_arora_example_minimizer = 1.386511
_arora_example_minimum = _arora_example_f(np.array([_arora_example_minimizer]))

def test_equally_spaced_line_search():
    x0 = 0.0
    d = np.array([+1.0])
    alpha = equal_interval_search(_arora_example_f, d, x0, DEFAULT_STRATEGY_FUNCTIONS, tol=1e-6)
    minimum = _arora_example_f(alpha * d)
    assert abs(minimum - _arora_example_minimum) <= 1e-6
