from scipy.optimize import minimize as scipy_min

from examples.lagrangian_problems.mechanical_project.constants import initial_guess
from examples.lagrangian_problems.mechanical_project.constraints import g7
from examples.lagrangian_problems.mechanical_project.linear_system import f


print(scipy_min(f, initial_guess, constraints=[{
    'type': 'ineq',
    'fun': lambda x:-g7(x)  # Constraints on scipy are g >= 0.0 instead of g <= 0.0
}]))
