import numpy as np



def default_grad_f_approx(f, x, h=1e-6):
    '''
    approximate grad(f) using finite differences for each term:

    f'(x) = f(x + h) - f(x)
            ---------------
                   h
    '''
    if type(x) == float:
        x = np.array([x])

    approx_grad = np.zeros(shape=(len(x),))
    f0 = f(x)
    for i in range(len(approx_grad)):
        x[i] += h
        approx_grad[i] = (f(x) - f0) / h
        x[i] -= h
    return approx_grad
