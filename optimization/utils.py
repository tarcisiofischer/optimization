import numpy as np

def build_penalized_function(
    f,
    n_vars,
    equality_constraints,
    inequality_constraints,
    phi,
    dfdx,
    dequality_constraintsdx,
    dinequality_constraintsdx,
):
    n_equality_constraints = len(equality_constraints)
    n_inequality_constraints = len(inequality_constraints)
    n_constraints = n_equality_constraints + n_inequality_constraints

    if type(phi) != list:
        phi = [phi] * n_constraints

    def P(x):
        result = f(x)
        for i in range(n_equality_constraints):
            phi_i = phi[i]
            h_i = equality_constraints[i](x)
            result += phi_i * (h_i ** 2)
        for j in range(n_inequality_constraints):
            phi_j = phi[n_equality_constraints + j]
            g_j = inequality_constraints[j](x)
            g_j_plus = max(0, g_j)
            result += phi_j * (g_j_plus ** 2)

        return result

    def grad_P(x):
        dL_dx = dfdx(x)
        for i in range(n_equality_constraints):
            phi_i = phi[i]
            grad_h_i = dequality_constraintsdx[i](x)
            dL_dx += phi_i * grad_h_i
        for j in range(n_inequality_constraints):
            phi_j = phi[n_equality_constraints + j]
            dg_jdx = dinequality_constraintsdx[j](x)
            grad_g_j_plus = np.array([max(0.0, dg_jdx_i) for dg_jdx_i in dg_jdx])
            dL_dx += phi_j * grad_g_j_plus

        return dL_dx

    return P, grad_P


def build_augmented_lagrangian(
    f,
    n_vars,
    equality_constraints,
    inequality_constraints,
    phi,
    dfdxx,
    dequality_constraintsdxx,
    dinequality_constraintsdxx,
):
    n_equality_constraints = len(equality_constraints)
    n_inequality_constraints = len(inequality_constraints)
    n_constraints = n_equality_constraints + n_inequality_constraints

    if type(phi) != list:
        phi = [phi] * n_constraints

    def L(x):
        xx = x[0:n_vars]
        lamb = x[n_vars:n_vars + n_equality_constraints]
        mi = x[n_vars + n_equality_constraints:n_vars + n_constraints + n_inequality_constraints]

        result = f(xx)
        for i in range(n_equality_constraints):
            phi_i = phi[i]
            lamb_i = lamb[i]
            h_i = equality_constraints[i](xx)

            result += lamb_i * h_i
            result += (phi_i / 2.0) * (h_i ** 2)
        for j in range(n_inequality_constraints):
            phi_j = phi[n_equality_constraints + j]
            mi_j = mi[j]
            g_j = inequality_constraints[j](xx)
            g_j_plus = max(-mi_j / phi_j, g_j)

            result += mi_j * g_j_plus
            result += (phi_j / 2.0) * (g_j_plus ** 2)

        return result

    def grad_L(x):
        xx = x[0:n_vars]
        lamb = x[n_vars:n_vars + n_equality_constraints]
        mi = x[n_vars + n_equality_constraints:n_vars + n_constraints + n_inequality_constraints]

        dL_dxx = dfdxx(xx)
        grad_eq = []
        for i in range(n_equality_constraints):
            phi_i = phi[i]
            lamb_i = lamb[i]
            h_i = equality_constraints[i](xx)
            grad_h_i = dequality_constraintsdxx[i](xx)

            dL_dxx += lamb_i * grad_h_i
            dL_dxx += phi_i * h_i * grad_h_i

            grad_eq.append(h_i)
        grad_ineq = []
        for j in range(n_inequality_constraints):
            phi_j = phi[n_equality_constraints + j]
            mi_j = mi[j]
            g_j = inequality_constraints[j](xx)
            dg_jdxx = dinequality_constraintsdxx[j](xx)
            g_j_plus = max(-mi_j / phi_j, g_j)
            grad_g_j_plus = np.array([max(0.0, dg_jdxx_i) for dg_jdxx_i in dg_jdxx])

            dL_dxx += mi_j * grad_g_j_plus
            dL_dxx += phi_j * g_j_plus * grad_g_j_plus

            grad_ineq.append(g_j_plus)

        return np.r_[dL_dxx, grad_eq, grad_ineq]

    return L, grad_L
