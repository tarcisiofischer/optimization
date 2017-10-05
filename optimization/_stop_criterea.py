import logging

import numpy as np


logger = logging.getLogger(__name__)

def default_stop_criterea(f, x, it, strategy_functions_dict, max_iter=500, tol=1e-2):
    gf = strategy_functions_dict['grad_f'](f, x)

    if it >= max_iter:
        logger.info("Finished with maximum number of iterations (%s)" % (max_iter,))
        logger.info("Gradient was %s" % (gf,))
        return True

    if np.allclose(gf, 0.0, atol=tol):
        logger.info("Finished with small gradient (%s)" % (gf,))
        logger.info("Number of iterations = %s" % (it,))
        return True

    logger.info("Iteration finished with grad f = %s. Will continue." % (gf,))
    return False


class residual_based_stop_criterea(object):

    def __init__(self):
        self._last_f_x = None


    def __call__(self, f, x, it, strategy_functions_dict, max_iter=500, tol=1e-4):
        f_x = f(x)

        if self._last_f_x is None:
            self._last_f_x = f_x
            return False

        diff = abs(self._last_f_x - f_x)
        if it >= max_iter:
            logger.info("Finished with maximum number of iterations (%s)" % (max_iter,))
            logger.info("Absolute difference was %s" % (diff,))
            return True

        if np.all(diff <= tol):
            logger.info("Finished with absolute difference of [%s]" % (diff,))
            logger.info("Number of iterations = %s" % (it,))
            return True

        self._last_f_x = f_x
        logger.info("Iteration finished with abs diff = %s. Will continue." % (diff,))
        return False
