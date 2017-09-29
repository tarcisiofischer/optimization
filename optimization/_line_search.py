import logging

import numpy as np


logger = logging.getLogger(__name__)



def equal_interval_search(f, d, x0, strategy_functions_dict, h=1.0, r=0.5, tol=1e-2):
    '''
    Equal Interval Search based on algorithm on Arora's book.
    '''
    if type(x0) == float:
        x0 = np.array([x0])

    # Phase I: Initial Bracketing of Minimum
    ff = lambda alpha: f(x0 + alpha * d)
    q = 0
    old_ff = ff(q * h)
    new_ff = ff((q + 1) * h)

    logger.info("Start at %s" % (old_ff,))
    logger.info('direction = %s' % (d,))
    while old_ff > new_ff:
        q += 1
        old_ff = new_ff
        new_ff = ff((q + 1) * h)
    alpha = (q - 1) * h

    # Phase II: Reducing the Interval of Uncertainty
    gf = strategy_functions_dict['grad_f'](ff, alpha)
    logger.info('alpha = %s' % (alpha,))
    logger.info('f(x + alpha*d) = %s' % (ff(alpha),))
    logger.info('grad f = %s' % (gf,))
    logger.info('')
    if abs(gf) < tol:
        return alpha
    else:
        return alpha + equal_interval_search(f, d, x0 + alpha * d, strategy_functions_dict, r * h, r, tol)


def armijo_backtracking_line_search(
    f,
    d,
    x0,
    strategy_functions_dict,
    alpha_i=1.0,
    tau=0.5,
    beta=0.01
    ):
    '''
    Will not find the minimum f(x), but an alpha such that f(alpha) < f(x0).
    beta controls the minimum desired decrease value on the function.
    tau controls the step break factor (Must be between (0.0, 1.0)).
    alpha_i is an initial guess for increment alpha.
    '''
    if type(x0) == float or type(x0) == np.float64:
        x0 = np.array([x0])

    assert 0.0 < beta < 1.0, 'beta must be in range (0.0, 1.0)'
    assert 0.0 < tau < 1.0, 'tau must be in range (0.0, 1.0)'

    alpha = alpha_i
    f0 = f(x0)
    gf = strategy_functions_dict['grad_f'](f, x0)
    f_alpha = f(x0 + alpha * d)
    beta_gf_d = beta * np.dot(gf, d)

    logger.info('Starting with alpha = %s' % (alpha,))
    logger.info('f0 = %s' % (f0,))
    while f_alpha > f0 + alpha * beta_gf_d:
        alpha = tau * alpha
        f_alpha = f(x0 + alpha * d)
        logger.info('alpha = %s' % (alpha,))
        logger.info('f_alpha = %s' % (f_alpha,))

    return x0 + alpha * d
