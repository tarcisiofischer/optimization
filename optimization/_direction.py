def steepest_descent(f, x, strategy_functions_dict):
    return -strategy_functions_dict['grad_f'](f, x)
