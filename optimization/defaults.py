from ._common import default_grad_f_approx
from ._direction import steepest_descent
from ._line_search import armijo_backtracking_line_search
from ._stop_criterea import default_stop_criterea

DEFAULT_STRATEGY_FUNCTIONS = {
    'grad_f': default_grad_f_approx,
    'stop_criterea': default_stop_criterea,
    'compute_direction': steepest_descent,
    'compute_step': armijo_backtracking_line_search,
    'plot_helper': lambda *args, **kwargs: None,
}
