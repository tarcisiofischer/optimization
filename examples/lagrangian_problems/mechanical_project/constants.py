import numpy as np

L = 200.  # [mm]
b = 10.  # [mm]
h_ini = 40.  # [mm]
E = 2.0e+5  # [N/mm2]
V_max = 2.4e+5  # [mm3]
h_min = 1.  # [mm]
h_max = 80.  # [mm]
P_y_4 = -200.  # [N]
initial_guess = np.array([h_ini, h_ini, h_ini])
