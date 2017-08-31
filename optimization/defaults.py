from ._common import default_grad_f_approx
from ._direction import steepest_descent
from ._line_search import equal_interval_search
from ._stop_criterea import default_stop_criterea

DEFAULT_STRATEGY_FUNCTIONS = {
    'grad_f': default_grad_f_approx,
    'stop_criterea': default_stop_criterea,
    'compute_direction': steepest_descent,
    'compute_step': equal_interval_search,
}
