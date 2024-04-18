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
import pandas as pd
from acados_template import latexify_plot


def plot_crop(t, u_max, u_min, U, X_true, X_est=None, Y_measured=None, energy_price_array=None, photoperiod_array=None, eod_array=None, min_DLI=None, max_DLI=None, latexify=False, plt_show=True,
               plot_all=False, states_labels=[], first_plot=[], X_true_label=None):
    if latexify:
        latexify_plot()
    if len(states_labels) == 0:
        raise IndexError('No state labels provided')

    WITH_ESTIMATION = X_est is not None and Y_measured is not None

    days = t / (60*60)
    if plot_all:
        total_plots = states_labels
    else:
        total_plots = first_plot
    subplot_rows = len(total_plots) + 2  # +2 for price and input plots

    fig, axs = plt.subplots(subplot_rows, 1, figsize=(10, min(subplot_rows * 3, 9)))

    # Plotting the energy prices
    axs[0].step(days, np.append([energy_price_array[0]], energy_price_array), where='post', label=X_true_label if X_true_label else 'Energy Price', color='r')
    axs[0].set_ylabel('$price$')
    axs[0].set_xlabel('$t$')
    axs[0].grid()

    # Plotting the input
    axs[1].step(days[:-1], U, where='post', label=X_true_label if X_true_label else 'Input', color='r')
    axs[1].hlines([u_max, u_min], days[0], days[-1], linestyles='dashed', colors=['g', 'b'], alpha=0.7)
    axs[1].set_ylabel('$u$')
    axs[1].set_xlabel('$t$')
    axs[1].grid()
    # Adding background color based on photoperiod

    for i in range(len(days)-1):
        if photoperiod_array[i] > 0:
            color = 'red'
        else:
            color = 'green'
        start = days[i]
        
        end = days[i + 1]
        axs[1].fill_betweenx([0, u_max], start, end, color=color,step='pre', alpha=0.08)
    #axs[1].fill_betweenx([0, u_max], end, end, color=color,step='pre', alpha=0.08)

    # Plotting states
    for j, label in enumerate(total_plots):
        idx = states_labels.index(label)
        axs[j+2].plot(days, X_true[:, idx], label='True state')
        if label == '$DLI_u$' and min_DLI is not None:
            axs[j+2].hlines([min_DLI, max_DLI], days[0], days[-1], linestyles='dashed', colors=['orange', 'purple'], alpha=0.7)
        if WITH_ESTIMATION:
            axs[j+2].plot(days_mhe, X_est[:, idx], '--', label='Estimated')
            axs[j+2].plot(days, Y_measured[:, idx], 'x', label='Measured')
        axs[j+2].set_ylabel(label)
        axs[j+2].set_xlabel('$t$')
        axs[j+2].legend(loc=1)
        axs[j+2].grid()

    plt.subplots_adjust(hspace=0.5)
    if plt_show:
        plt.show()
def generate_energy_price(N_horizon, Nsim = None):
    def f(i):
        return 10 + i % 5
    energy_price_array = np.ones((N_horizon))
    for i in range(N_horizon):

        energy_price_array[i] = f(i)

    #energy_price_array = energy_price_array / average_price
    print('-'*20)
    print('The energy price array is: ',energy_price_array)
    print('-'*20)
    if Nsim is None:
        return energy_price_array, None
    closed_loop_energy_price_array = np.ones((Nsim))
    for i in range(Nsim):
        closed_loop_energy_price_array[i] = f(i)
    return energy_price_array, closed_loop_energy_price_array

def generate_photoperiod_values(photoperiod_length: int, darkperiod_length: int, N_horizon: int, Nsim: int = None):
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
    if Nsim is None:
        
        
        # Continue appending values to the sequence until its length reaches N_horizon
        while len(sequence) < N_horizon:
            # Append 1s for the photoperiod
            sequence += [0] * photoperiod_length
            # Ensure the sequence does not exceed N_horizon
            if len(sequence) >= N_horizon:
                break
            # Append 0s for the darkperiod
            sequence += [1] * darkperiod_length
        
        # Trim the sequence if it exceeds N_horizon
        sequence = sequence[:N_horizon]
    
    
        # Convert the sequence to a NumPy array and return it
        return np.array(sequence), None

    while len(sequence) < Nsim + N_horizon:
        # Append 1s for the photoperiod
        sequence += [0] * photoperiod_length
        # Ensure the sequence does not exceed N_horizon
        if len(sequence) >= Nsim + N_horizon:
            break
        # Append 0s for the darkperiod
        sequence += [1] * darkperiod_length
    
    # Trim the sequence if it exceeds N_horizon
    sequence = sequence[:Nsim + N_horizon]


    # Convert the sequence to a NumPy array and return it
    return np.array(sequence[:N_horizon]), np.array(sequence)
    
def generate_end_of_day_array(photoperiod_length: int, darkperiod_length: int, N_horizon: int, Nsim: int = None):
    sequence = []
    if Nsim is None:
        while len(sequence) < N_horizon:
            sequence += [0] * (photoperiod_length - 1)
            sequence += [1]
            if len(sequence) >= N_horizon:
                break
            sequence += [0] * darkperiod_length
        sequence = sequence[:N_horizon]

        return np.array(sequence), None
    while len(sequence) < Nsim + N_horizon:
        sequence += [0] * (photoperiod_length - 1)
        sequence += [1]
        if len(sequence) >= Nsim + N_horizon:
            break
        sequence += [0] * darkperiod_length
    sequence = sequence[:Nsim + N_horizon]
    return np.array(sequence[:N_horizon]), np.array(sequence)
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
    if hasattr(ocp.constraints, 'lh'):
        print(f"Lower bound on algebraic variable (lh): {ocp.constraints.lh}")
    if hasattr(ocp.constraints, 'uh'):
        print(f"Upper bound on algebraic variable (uh): {ocp.constraints.uh}")
    
    # Additional details if needed
    # This part can be extended based on what specific details are crucial for your debugging process.
def fetch_electricity_prices(file_path, length, length_sim = None, start_datetime='2016-01-01 Kl. 01-02', column='NO1'):

    data = pd.read_csv(file_path, sep=';', index_col='Dato/klokkeslett')

    # Finn starttime in the dataset
    if start_datetime not in data.index:
        raise ValueError("Start-date does not exist in the dataset (2016-01-01 01-02 is the earliest and 2014-04-15 23-00 is the latest).")

    # Find indeks for startdate
    start_index = data.index.get_loc(start_datetime)

    # Kontroller at det er nok data tilgjengelig fra startpunktet
    if start_index + length > len(data):
        raise ValueError("Not enough available length from start to end date.")

    # Hent data fra den angitte kolonnen og lengden
    prices = data.loc[data.index[start_index:start_index + length], column].values

    # Konverter til numpy array og returner
    if length_sim is None:
        return np.array(prices, dtype=float), None
    start_index = data.index.get_loc(start_datetime)
    # Kontroller at det er nok data tilgjengelig fra startpunktet
    if start_index + length_sim + length > len(data):
        raise ValueError("Not enough available length from start to end date (closed loop).")
    prices_sim = data.loc[data.index[start_index:start_index + length_sim + length], column].values
    return prices, prices_sim