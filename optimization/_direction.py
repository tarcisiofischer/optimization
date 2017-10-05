def steepest_descent(f, x, strategy_functions_dict):
    return -strategy_functions_dict['grad_f'](f, x)


class conjugate_gradient(object):

    def __init__(self):
        self._last_grad = None

    def __call__(self, f, x, strategy_functions_dict):
        import numpy as np

        if self._last_grad is None:
            self._last_grad = -strategy_functions_dict['grad_f'](f, x)
            return self._last_grad

        c_0 = self._last_grad
        c_1 = strategy_functions_dict['grad_f'](f, x)
        betha = (np.linalg.norm(c_1) / np.linalg.norm(c_0)) ** 2.0

        self._last_grad = c_1
        return -c_1 - betha * (c_0)
