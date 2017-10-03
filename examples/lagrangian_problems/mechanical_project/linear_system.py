from numpy.linalg.linalg import LinAlgError

from constants import b, E, L, P_y_4
from examples.lagrangian_problems.mechanical_project.constants import h_ini
import numpy as np


coefs = np.array([
    [
        (12.*E / L ** 3),
        (6.*E / L ** 2),
        (-12.*E / L ** 3),
        (6.*E / L ** 2),
        0.0,
        0.0
    ],
    [
        (6.*E / L ** 2),
        (4.*E / L),
        (-6.*E / L ** 2),
        (2.*E / L),
        0.0,
        0.0
    ],
    [
        (-12.*E / L ** 3),
        (-6.*E / L ** 2),
        (12.*E / L ** 3),
        (6.*E / L ** 2),
        (-12.*E / L ** 3),
        (6.*E / L ** 2)
    ],
    [
        (6.*E / L ** 2),
        (2.*E / L),
        (6.*E / L ** 2),
        (4.*E / L),
        (-6.*E / L ** 2),
        (2.*E / L)
    ],
    [
        0.0,
        0.0,
        (-12.*E / L ** 3),
        (-6.*E / L ** 2),
        (12.*E / L ** 3),
        (-6.*E / L ** 2)
    ],
    [
        0.0,
        0.0,
        (6.*E / L ** 2),
        (2.*E / L),
        (-6.*E / L ** 2),
        (4.*E / L)
    ]
])

def K(x):
    h1, h2, h3 = x[0], x[1], x[2]

    I_1 = b * h1 ** 3.0 / 12.0
    I_2 = b * h2 ** 3.0 / 12.0
    I_3 = b * h3 ** 3.0 / 12.0
    I = np.array([
        [
            (I_1 + I_2),
            (I_2 - I_1),
            I_2,
            I_2,
            0.0,
            0.0
        ],
        [
            (I_2 - I_1),
            (I_1 + I_2),
            I_2,
            I_2,
            0.0,
            0.0
        ],
        [
            I_2,
            I_2,
            (I_2 + I_3),
            (I_3 - I_2),
            I_3,
            I_3
        ],
        [
            I_2,
            I_2,
            (I_3 - I_2),
            (I_2 + I_3),
            I_3,
            I_3
        ],
        [
            0.0,
            0.0,
            I_3,
            I_3,
            I_3,
            I_3
        ],
        [
            0.0,
            0.0,
            I_3,
            I_3,
            I_3,
            I_3
        ]
    ])
    return coefs * I


def u(x):
    F = np.array([
        0.0,
        0.0,
        0.0,
        0.0,
        P_y_4,
        0.0,
    ])
    try:
        K_i = np.linalg.inv(K(x))
    except:
        return np.ones_like(F) * 10000.0
    return np.dot(K_i, F)


def dudx(x):
    h1, h2, h3 = x[0], x[1], x[2]

    dI_1_dh1 = (b * h1 ** 2.0) / 4.0
    dKdh1 = coefs * np.array([
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]) * dI_1_dh1

    dI_2_dh2 = (b * h2 ** 2.0) / 4.0
    dKdh2 = coefs * np.array([
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, -1.0, 0.0, 0.0],
        [1.0, 1.0, -1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]) * dI_2_dh2

    dI_3_dh3 = (b * h3 ** 2.0) / 4.0
    dKdh3 = coefs * np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    ]) * dI_3_dh3

    K_i = np.linalg.inv(K(x))
    u_x = u(x)
    return np.array([
        np.dot(K_i, (0.0 - np.dot(dKdh1, u_x))),
        np.dot(K_i, (0.0 - np.dot(dKdh2, u_x))),
        np.dot(K_i, (0.0 - np.dot(dKdh3, u_x)))
    ])


def dfdx_adjoint(x):
    h1, h2, h3 = x[0], x[1], x[2]

    dI_1_dh1 = (b * h1 ** 2.0) / 4.0
    dKdh1 = coefs * np.array([
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]) * dI_1_dh1

    dI_2_dh2 = (b * h2 ** 2.0) / 4.0
    dKdh2 = coefs * np.array([
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, -1.0, 0.0, 0.0],
        [1.0, 1.0, -1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]) * dI_2_dh2

    dI_3_dh3 = (b * h3 ** 2.0) / 4.0
    dKdh3 = coefs * np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    ]) * dI_3_dh3

    df_du = np.array([
        0.0,
        0.0,
        0.0,
        0.0,
        0.5 * P_y_4,
        0.0
    ])
    K_t = K(x).T
    lamb = np.linalg.solve(K_t, df_du)

    df_dh = 0.0
    dF_dh = 0.0
    u_x = u(x)
    return np.array([
        df_dh + np.dot(dF_dh, lamb) - np.dot(np.dot(dKdh1, u_x), lamb),
        df_dh + np.dot(dF_dh, lamb) - np.dot(np.dot(dKdh2, u_x), lamb),
        df_dh + np.dot(dF_dh, lamb) - np.dot(np.dot(dKdh3, u_x), lamb),
    ])
