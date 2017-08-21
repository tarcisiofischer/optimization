from optimization._common import default_grad_f_approx
import numpy as np


def test_grad_f():
    grad_f = default_grad_f_approx

    def f(x):
        return x[0] ** 2 + 2 * x[1] - x[0] * x[1]

    def exact_grad_f(x):
        return np.array([
            2 * x[0] - x[1],
            2 - x[0]
        ])

    for x in [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([-1.0, -1.0]),
    ]:
        assert np.allclose(grad_f(f, x), exact_grad_f(x), rtol=1e-4, atol=1e-6)
