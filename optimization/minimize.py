from optimization.defaults import DEFAULT_STRATEGY_FUNCTIONS


def minimize(f, x0, strategy_functions_dict=None):
    '''
    '''
    if strategy_functions_dict is None:
        strategy_functions_dict = DEFAULT_STRATEGY_FUNCTIONS

    x = x0
    it = 0
    while not strategy_functions_dict['stop_criterea'](f, x, it, strategy_functions_dict):
        d = strategy_functions_dict['compute_direction'](f, x, strategy_functions_dict)
        alpha = strategy_functions_dict['compute_step'](f, d, x, strategy_functions_dict)
        x += alpha * d
        it += 1
    return x
