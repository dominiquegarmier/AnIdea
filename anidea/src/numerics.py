# AnIdea (c) 2019 Dominique F. Garmier All Rights Reserved
# Version: pre2.1
# --------------------------------------------------------

import numpy as np

def PendulumEulerMethod(x0, v0, t0, omega, t_range, stepsize):
    '''returns x_plot, t_plot'''

    time_plot = t0 + np.linspace(0, t_range, int(t_range/stepsize), endpoint=False)
    x_plot = np.array([])

    x_i = x0
    v_i = v0

    for i in range(int(t_range/stepsize)):

        x_plot = np.append(x_plot, x_i)

        a_i = -omega * np.sin(x_i)

        x_i1 = x_i + stepsize * v_i 
        v_i1 = v_i + stepsize * a_i

        x_i = x_i1
        v_i = v_i1

    return x_plot, time_plot

def PendulumRK4(x0, v0, t0, omega, t_range, stepsize):
    '''returns x_plot, t_plot'''

    time_plot = t0 + np.linspace(0, t_range, int(t_range/stepsize), endpoint=False)
    x_plot = np.array([])

    x_i = x0
    v_i = v0

    for i in range(int(t_range/stepsize)):

        x_plot = np.append(x_plot, x_i)
        
        k_x1 = stepsize * v_i
        k_v1 = -stepsize * omega * np.sin(x_i)

        k_x2 = stepsize * (v_i + 0.5*k_v1)
        k_v2 = -stepsize * omega * np.sin(x_i + 0.5*k_x1)

        k_x3 = stepsize * (v_i + 0.5*k_v2)
        k_v3 = -stepsize * omega * np.sin(x_i + 0.5*k_x2)

        k_x4 = stepsize * (v_i + k_v3)
        k_v4 = -stepsize * omega * np.sin(x_i + k_x3)
        
        x_i1 = x_i + (1/6)*(k_x1 + 2*k_x2 + 2*k_x3 + k_x4)
        v_i1 = v_i + (1/6)*(k_v1 + 2*k_v2 + 2*k_v3 + k_v4)

        x_i = x_i1
        v_i = v_i1

    return x_plot, time_plot