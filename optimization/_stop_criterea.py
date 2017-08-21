import logging

import numpy as np


logger = logging.getLogger(__name__)

def default_stop_criterea(f, x, it, strategy_functions_dict, max_iter=500, tol=1e-6):
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
