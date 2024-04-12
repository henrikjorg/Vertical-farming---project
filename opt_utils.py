#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

import matplotlib.pyplot as plt
import numpy as np
from acados_template import latexify_plot


def plot_crop(t, u_max,u_min, U, X_true, X_est=None, Y_measured=None, latexify=False, plt_show=True, X_true_label=None):
    """
    Params:
        t: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
        X_est: arrray with shape (N_sim-N_mhe, nx)
        Y_measured: array with shape (N_sim, ny)
        latexify: latex style plots
    """

    if latexify:
        latexify_plot()

    WITH_ESTIMATION = X_est is not None and Y_measured is not None

    N_sim = X_true.shape[0]
    nx = X_true.shape[1]

    Tf = t[N_sim-1]
    Ts = t[1] - t[0]

    if WITH_ESTIMATION:
        N_mhe = N_sim - X_est.shape[0]
        t_mhe = np.linspace(N_mhe * Ts, Tf, N_sim-N_mhe)
        days_mhe = t_mhe / (60*60*24)

    plt.subplot(nx+1, 1, 1)
    days = t / (60*60*24)
    line, = plt.step(days, np.append([U[0]], U))
    if X_true_label is not None:
        line.set_label(X_true_label)
    else:
        line.set_color('r')

    plt.ylabel('$u$')
    plt.xlabel('$t$')
    plt.hlines(u_max, days[0], days[-1], linestyles='dashed', alpha=0.7)
    plt.hlines(u_min, days[0], days[-1], linestyles='dashed', alpha=0.7)
    if u_min < 0:
        plt.ylim([1.2*u_min, 1.2*u_max])
    elif u_min == 0:
        plt.ylim([-10, 1.2*u_max])
    else:
        plt.ylim([0.8*u_min, 1.2*u_max])
    plt.xlim(days[0], days[-1])
    plt.grid()

    states_lables = ['$X_ns$', '$X_s$', '$FW_per_plant$', '$t$']

    for i in range(nx):
        plt.subplot(nx+1, 1, i+2)
        line, = plt.plot(days, X_true[:, i], label='true')
        if X_true_label is not None:
            line.set_label(X_true_label)

        if WITH_ESTIMATION:
            plt.plot(days_mhe, X_est[:, i], '--', label='estimated')
            plt.plot(days, Y_measured[:, i], 'x', label='measured')

        plt.ylabel(states_lables[i])
        plt.xlabel('$t$')
        plt.grid()
        plt.legend(loc=1)
        plt.xlim(t[0], days[-1])

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    if plt_show:
        plt.show()
def generate_energy_price(N_horizon):
    
    energy_price_array = np.ones((N_horizon))
    sum = 0
    for i in range(N_horizon):
        if i % 24 > 16:
            energy_price_array[i] = 10000000000
        else:
            energy_price_array[i] = 10 + i % 5
        
        sum += energy_price_array[i] 
    average_price = sum/N_horizon
    #energy_price_array = energy_price_array / average_price
    print('-'*20)
    print('The energy price array is: ',energy_price_array)
    print('-'*20)
    return energy_price_array

def generate_photoperiod_values(photoperiod: int, darkperiod: int, N_horizon: int) -> np.array:
    """
    Generates an array representing a sequence of light and dark periods over a specified horizon.
    
    The generated array alternates between values of 1 (representing light periods) and 0 (representing dark periods),
    starting with a light period. The sequence continues, alternating between the specified durations of light and dark periods,
    until the length of the array reaches the specified horizon.
    
    Parameters:
    - photoperiod (int): The duration of the light period within the cycle.
    - darkperiod (int): The duration of the dark period within the cycle.
    - N_horizon (int): The total length of the horizon over which to generate the sequence.
    
    Returns:
    - np.array: An array of length N_horizon, with values alternating between 1s and 0s according to the specified photoperiod and darkperiod.
    """
    # Initialize an empty list to hold the sequence of light and dark period values
    sequence = []
    
    # Continue appending values to the sequence until its length reaches N_horizon
    while len(sequence) < N_horizon:
        # Append 1s for the photoperiod
        sequence += [1] * photoperiod
        # Ensure the sequence does not exceed N_horizon
        if len(sequence) >= N_horizon:
            break
        # Append 0s for the darkperiod
        sequence += [1] * darkperiod
    
    # Trim the sequence if it exceeds N_horizon
    sequence = sequence[:N_horizon]
    
    # Convert the sequence to a NumPy array and return it
    return np.array(sequence)

def print_ocp_setup_details(ocp):
    print("Optimal Control Problem Setup Details:")

    # Printing cost function details
    print("\nCost Function:")
    print(f"Cost type: {ocp.cost.cost_type}")
    if ocp.cost.cost_type == 'EXTERNAL':
        print("External cost function setup not directly visible.")
    else:
        print(f"W = [Q,,0; 0, R] matrix (state weights): {ocp.cost.W}")
        if hasattr(ocp.cost, 'Qe'):
            print(f"Qe matrix (terminal state weights): {ocp.cost.W_e}")

    # Printing constraints
    print("\nConstraints:")
    print(f"Lower bound on control inputs (lbu): {ocp.constraints.lbu}")
    print(f"Upper bound on control inputs (ubu): {ocp.constraints.ubu}")
    if hasattr(ocp.constraints, 'idxbu'):
        print('-'*5)
        print(f"Indices of bounds on control inputs (idxbu): {ocp.constraints.idxbu}")
        print('-'*10)
    if hasattr(ocp.constraints, 'x0'):
        print('-'*20)
        print(f"Initial condition constraints (x0): {ocp.constraints.x0}")
        print('-'*30)
    if hasattr(ocp.constraints, 'lbx'):
        print(f"Lower bound on states (lbx): {ocp.constraints.lbx}")
    if hasattr(ocp.constraints, 'ubx'):
        print(f"Upper bound on states (ubx): {ocp.constraints.ubx}")
    
    # Additional details if needed
    # This part can be extended based on what specific details are crucial for your debugging process.
