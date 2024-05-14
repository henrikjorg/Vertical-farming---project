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
import sys
 
# setting path
sys.path.append('../VF-SIMULATION')
from data.utils import fetch_electricity_prices
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from acados_template import latexify_plot
import casadi as ca
from utilities import *

def plot_crop(t, u_max, u_min, U, X_true, X_est=None, Y_measured=None, energy_price_array=None, photoperiod_array=None,eod_array=None, Z_true = None, Z_labels = None, min_DLI=None, max_DLI=None, latexify=False, plt_show=True,
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

    # First figure setup
    fig1, axs1 = plt.subplots(subplot_rows, 1, figsize=(10, min(subplot_rows * 3, 9)))

    # Second figure setup, check if Z_true is provided
    if Z_labels is not None:
        subplot_rows_z = len(Z_labels) + 2
        fig2, axs2 = plt.subplots(subplot_rows_z, 1, figsize=(10, min(subplot_rows_z * 3, 9)))

    # Function to plot energy price and input
    def plot_price_input(axs, days, energy_price_array, U, u_max, u_min, photoperiod_array, label_suffix=''):
        if label_suffix == ' with Z':
            axs[0].step(days[:-1], energy_price_array, where='post', label='Energy Price' + label_suffix, color='r')
        else:
            axs[0].step(days, np.append([energy_price_array[0]], energy_price_array), where='post', label='Energy Price' + label_suffix, color='r')

        axs[0].set_ylabel('$price$')
        axs[0].set_xlabel('$t$')
        axs[0].grid()
        if U.shape[0] > 1:
            if label_suffix==' with Z':
                axs[1].step(days[:-1], U, where='post', label='Input' + label_suffix, color='r')
            else:
                axs[1].step(days[:-1], U, where='post', label='Input' + label_suffix, color='r')
                
        else:

            axs[1].step(days, np.append(0, U), where='pre', label='Input' + label_suffix, color='r')
               
        axs[1].hlines([u_max, u_min], days[0], days[-2], linestyles='dashed', colors=['g', 'b'], alpha=0.7)
        axs[1].set_ylabel('$u$')
        axs[1].set_xlabel('$t$')
        axs[1].grid()
        if label_suffix == ' with Z':
            r = len(days) -2
        else:
            r = len(days)-1
        for i in range(r):
            color = 'red' if photoperiod_array[i] > 0 else 'green'
            start = days[i]
            
            end = days[i + 1]
            axs[1].fill_betweenx([0, u_max], start, end, color=color, step='pre', alpha=0.08)

    # Plot energy price and input for both figures
    plot_price_input(axs1, days, energy_price_array, U, u_max, u_min, photoperiod_array)
    if Z_labels is not None:
        plot_price_input(axs2, days, energy_price_array, U, u_max, u_min, photoperiod_array, ' with Z')

    # Plot states for the first figure
    for j, label in enumerate(total_plots):
        idx = states_labels.index(label)
        axs1[j+2].plot(days, X_true[idx, :])
        if label == '$DLI_u$' and min_DLI is not None:
            axs1[j+2].hlines([min_DLI, max_DLI], days[0], days[-1], linestyles='dashed', colors=['orange', 'purple'], alpha=0.7)
        if label == '$DLI_u$':
            r = len(days)-1
            for i in range(r):
                color = 'red' if eod_array[i] == 0 else 'green'
                start = days[i]
                
                end = days[i + 1]
                axs1[j+2].fill_betweenx([0, max_DLI], start, end, color=color, step='pre', alpha=0.08)
        if WITH_ESTIMATION:
            axs1[j+2].plot(days_mhe, X_est[:, idx], '--', label='Estimated')
            axs1[j+2].plot(days, Y_measured[:, idx], 'x', label='Measured')
        axs1[j+2].set_ylabel(label)
        axs1[j+2].set_xlabel('$t$')
        axs1[j+2].grid()

    # Plot Z states for the second figure if provided
    if Z_labels is not None:
        for j, label in enumerate(Z_labels):
            if Z_true.shape[0] > 1:
                if label == '$z_1$':
                    r = len(days)-1
                    for i in range(r):
                        color = 'red' if eod_array[i] == 0 else 'green'
                        start = days[i]
                        
                        end = days[i + 1]
                        axs2[j+2].fill_betweenx([0, max_DLI], start, end, color=color, step='pre', alpha=0.08)
                axs2[j+2].plot(days[:-1], Z_true[:, j])
            else:
                axs2[j+2].plot(days, np.append(Z_true[:, j], Z_true[:, j]))

            axs2[j+2].set_ylabel(label)
            axs2[j+2].set_xlabel('$t$')
            axs2[j+2].grid()

    plt.subplots_adjust(hspace=0.5)
    if plt_show:
        plt.show()



def plot_crop_casadi(t, u_max, u_min, U, X_true,  energy_price_array=None, photoperiod_array=None,eod_array=None, min_DLI=None, max_DLI=None, latexify=False, plt_show=True,
               states_labels=[], labels_to_plot=[], min_points=[], max_points=[], end_mass=[], block=False, end_time=-1, title=''):
    if latexify:
        latexify_plot()
    if len(states_labels) == 0:
        raise IndexError('No state labels provided')


    days = t / (60*60)

    total_plots = labels_to_plot
    subplot_rows = len(total_plots) + 2  # +2 for price and input plots

    # First figure setup
    fig1, axs1 = plt.subplots(subplot_rows, 1, figsize=(10, min(subplot_rows * 3, 9)))

    # Function to plot energy price and input
    def plot_price_input(axs, days, energy_price_array, U, u_max, u_min, photoperiod_array, label_suffix=''):
        
        axs[0].step(days, np.append(energy_price_array, [energy_price_array[-1]]), where='post', label='Energy Price' + label_suffix, color='r')

        axs[0].set_ylabel('$price$')
        axs[0].set_xlabel('$t$')
        axs[0].grid()
        if U[0,:].shape[0] > 1:
            
            axs[1].step(days[:-1], U[0,:], where='post', label='Input' + label_suffix, color='r')
                
        else:

            axs[1].step(days, np.append(0, U[0,:]), where='pre', label='Input' + label_suffix, color='r')
               
        axs[1].hlines([u_max, u_min], days[0], days[-2], linestyles='dashed', colors=['g', 'b'], alpha=0.7)
        axs[1].set_ylabel('$Light Input$')
        axs[1].set_xlabel('$t$')
        axs[1].grid()
        
        r = len(days)-1
        for i in range(r):
            color = 'red' if eod_array[i] > 0 else 'green'
            start = days[i]
            
            end = days[i + 1]
            axs[1].fill_betweenx([0, u_max], start, end, color=color, step='pre', alpha=0.08)

        
    # Plot energy price and input for both figures
    plot_price_input(axs1, days, energy_price_array, U, u_max, u_min, photoperiod_array)
    # Plot states for the first figure
    for j, label in enumerate(total_plots):
        idx = states_labels.index(label)
        axs1[j+2].plot(days, X_true[idx, :])
        if label == 'DLI':
            axs1[j+2].hlines([min_DLI, max_DLI], days[0], days[-2], linestyles='dashed', colors=['g', 'b'], alpha=0.7)
            if len(min_points) > 0:
                for ind, y in min_points:
                    axs1[j+2].scatter(days[ind], y, marker='x', s=4)
                for ind, y in max_points:
                    axs1[j+2].scatter(days[ind], y, marker='x', s=4)
        if label == 'Fresh weight':
            axs1[j+2].scatter(days[end_time], end_mass, marker='x', s=6)
        axs1[j+2].set_ylabel(label)
        axs1[j+2].set_xlabel('$t$')
        axs1[j+2].grid()
    fig1.suptitle(title)
    plt.subplots_adjust(hspace=0.5)
    if plt_show:
        plt.show(block=block)


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

def generate_photoperiod_values(photoperiod_length: int, darkperiod_length: int, Nsim: int = None):
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
    
    while len(sequence) < Nsim :
        # Append 1s for the photoperiod
        sequence += [0] * photoperiod_length
        # Ensure the sequence does not exceed N_horizon
        if len(sequence) >= Nsim:
            break
        # Append 0s for the darkperiod
        sequence += [1] * darkperiod_length
    
    # Trim the sequence if it exceeds N_horizon
    sequence = sequence[:Nsim ]


    # Convert the sequence to a NumPy array and return it
    return np.array(sequence)
    
def generate_end_of_day_array(photoperiod_length: int, darkperiod_length: int, Nsim: int = None):
    sequence = []
    while len(sequence) < Nsim:
        sequence += [0] * (photoperiod_length-1)
        sequence += [1]
        if len(sequence) >= Nsim:
            break
        
        sequence += [0] * (darkperiod_length)
    sequence = sequence[:Nsim]
    return np.array(sequence)


def generate_start_of_night_array(photoperiod_length: int, darkperiod_length: int, Nsim: int = None):
    sequence = []
    
    while len(sequence) < Nsim :
        sequence += [0] * (photoperiod_length)
        
        if len(sequence) >= Nsim :
            break
        sequence += [1]
        sequence += [0] * (darkperiod_length-1)
    sequence = sequence[:Nsim]
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
    if hasattr(ocp.constraints, 'lh'):
        print(f"Lower bound on algebraic variable (lh): {ocp.constraints.lh}")

    if hasattr(ocp.constraints, 'uh'):
        print(f"Upper bound on algebraic variable (uh): {ocp.constraints.uh}")
    
    # Additional details if needed
    # This part can be extended based on what specific details are crucial for your debugging process.



def generate_data_dict(photoperiod_length=None, darkperiod_length=None, Nsim=None, shrink=False, N_horizon = None):
    # N_horizon only used if shrink = False
    data_dict = {}
    if shrink:
        data_dict['energy'] = fetch_electricity_prices('data/Spotprices_norway.csv', length=Nsim)
        data_dict['photoperiod'] = generate_photoperiod_values(photoperiod_length=photoperiod_length, darkperiod_length=darkperiod_length, Nsim=Nsim)
        data_dict['eod'] = generate_end_of_day_array(photoperiod_length=photoperiod_length, darkperiod_length=darkperiod_length, Nsim=Nsim)
        data_dict['son'] = generate_start_of_night_array(photoperiod_length=photoperiod_length, darkperiod_length=darkperiod_length, Nsim=Nsim)
    else:
        data_dict['energy'] = fetch_electricity_prices('data/Spotprices_norway.csv', length=Nsim+N_horizon)
        data_dict['photoperiod'] = generate_photoperiod_values(photoperiod_length=photoperiod_length, darkperiod_length=darkperiod_length, Nsim=Nsim+N_horizon)
        data_dict['eod'] = generate_end_of_day_array(photoperiod_length=photoperiod_length, darkperiod_length=darkperiod_length, Nsim=Nsim+N_horizon)
        data_dict['son'] = generate_start_of_night_array(photoperiod_length=photoperiod_length, darkperiod_length=darkperiod_length, Nsim=Nsim+N_horizon)
    return data_dict


def get_explicit_model(Crop, Env, is_rate_degradation=False, c_degr=400, ts=3600, photoperiod_length=24, saturation_curve_type='rectangular'):
    # Creating the states
    x_ns = ca.MX.sym('x_ns')
    x_s = ca.MX.sym('x_s')
    x_fw = ca.MX.sym('x_fw')
    DLI = ca.MX.sym('DLI')
    u_light = ca.MX.sym('u_light')
    u_prev = ca.MX.sym('u_prev')

    x = ca.vertcat(x_ns, x_s, x_fw, DLI)

    u_control = ca.vertcat(u_light, u_prev)

    nx = x.shape[0]
    nu = u_control.shape[0]
    # The model
    T_air = Env.T_air
    CO2_air = Env.CO2


    PAR_flux = u_light * 0.217

    # Calculate stomatal and aerodynamic conductances (reciprocals of resistances)
    LAI = SLA_to_LAI(SLA=Crop.SLA, c_tau=Crop.c_tau, leaf_to_shoot_ratio=Crop.leaf_to_shoot_ratio, X_s=x_s, X_ns=x_ns)
    g_stm = 1 / stomatal_resistance_eq(u_light)

    g_bnd = 1 / aerodynamical_resistance_eq(Env.air_vel, LAI=LAI, leaf_diameter=Crop.leaf_diameter)


    # Dynamics equations adapted for CasADi
    g_car = Crop.c_car_1 * T_air**2 + Crop.c_car_2 * T_air + Crop.c_car_3
    g_CO2 = 1 / (1 / g_bnd + 1 / g_stm + 1 / g_car)
    Gamma = Crop.c_Gamma * Crop.c_q10_Gamma ** ((T_air - 20) / 10)

    #ULM_degratation = exp(- (PPFD - last_u)**2/400**2)
    epsilon_biomass = Crop.c_epsilon * (CO2_air - Gamma) / (CO2_air + 2 * Gamma)
    if saturation_curve_type == 'rectangular':
        f_phot_max = (epsilon_biomass * PAR_flux * g_CO2 * Crop.c_w * (CO2_air - Gamma)) / (epsilon_biomass * PAR_flux + g_CO2 * Crop.c_w * (CO2_air - Gamma))
    elif saturation_curve_type == 'exponential':
        A_sat = g_CO2 * Crop.c_w * (CO2_air - Gamma)
        k_slope =  0.87* epsilon_biomass / A_sat
        f_phot_max = A_sat * (1 - np.exp(-k_slope * PAR_flux))
    
    if is_rate_degradation:
        f_phot_max *= ca.exp(-((u_light-u_prev)/c_degr)**2)
    
    #f_phot_max = 0.000006*PAR_flux
    f_phot = (1 - np.exp(-Crop.c_K * LAI)) * f_phot_max 
    f_resp = (Crop.c_resp_sht * (1 - Crop.c_tau) * x_s + Crop.c_resp_rt * Crop.c_tau * x_s) * Crop.c_q10_resp ** ((T_air - 25) / 10)
    dw_to_fw = (1 - Crop.c_tau) / (Crop.dry_weight_fraction * Crop.plant_density)
    r_gr = Crop.c_gr_max * x_ns / (Crop.c_gamma * x_s + x_ns + 0.001) * Crop.c_q10_gr ** ((T_air - 20) / 10)
    dX_ns = Crop.c_a * f_phot - r_gr * x_s - f_resp - (1 - Crop.c_beta) / Crop.c_beta * r_gr * x_s
    dX_s = r_gr * x_s

    f_expl_day = ca.vertcat(dX_ns,                               # Non-structural dry weight per m^2
                    dX_s,                                # Structural dry weight per m^2
                    (dX_ns + dX_s) * dw_to_fw ,           # Fresh weight of the shoot of one plant
                    u_light/(ts * photoperiod_length)#PPFD/(Ts*N_horizon),                 # Average PPFD per day
                    )
    f = ca.Function('f', [x, u_control], [f_expl_day], ['x', 'u_control'], ['ode'])
    intg_options = {
        'tf' : ts,
        'simplify' : True
        
    }

    dae = {
        'x': x,
        'p': u_control,
        'ode': f(x, u_control)
    }

    intg = ca.integrator('intg', 'rk', dae, intg_options)

    res = intg(x0=x, p= u_control)

    x_next_day = res['xf']

    # Simplifying API to (x,u) -> (x_next)
    # F simulates the function for one time step
    F = ca.Function('F', [x,u_control], [x_next_day], ['x', 'u_control'], ['x_next'])

    return F, nx, nu

def update_opti(opti, F,x,u,p,energy,current_time, end_time, N_horizon, l_end_mass, u_end_mass, min_DLI, max_DLI, u_min, u_max, current_state, current_energy):
    obj = ca.dot(u[0,:], energy)
    
     # Parameters for penalties
    penalty_weight = 1e6  # Adjust this weight to increase/decrease the penalty
    print(end_time)
    print('--'*20)
    # Penalty for being below lower bound
    penalty_below = ca.fmax(0, l_end_mass - x[2, end_time])
    # Penalty for exceeding upper bound
    penalty_above = ca.fmax(0, x[2, end_time] - u_end_mass)
    obj += penalty_weight * (penalty_below**2 + penalty_above**2)


    for k in range(end_time+1):
        if (current_time + k) % 24 == 0 and  k > 0:
            penalty_below_DLI = ca.fmax(0, min_DLI - x[3,k] + x[3,k-24])
            penalty_above_DLI = ca.fmax(0,  x[3,k] - x[3,k-24] - max_DLI)
            obj += penalty_weight* (penalty_below_DLI**2 + penalty_above_DLI**2)


    opti.minimize(obj)
    print(obj)
    opti.set_value(p, current_state)
    opti.set_value(energy, current_energy)
    opti.subject_to(x[:,0] == p)
    
    return opti
    """
    opti.subject_to()
    for k in range(N_horizon):
        opti.subject_to(x[:,k+1] == F(x[:,k], u[:,k]))
        opti.subject_to([u[0,k] <= u_max, u[0,k] >= u_min])

    # Setting u_prev
    opti.subject_to(u[1,0] == 0)
    for k in range(1, N_horizon):
        opti.subject_to(u[1, k] == u[0, k-1])
    opti.set_value(p, current_state)
    opti.set_value(energy, current_energy)
    opti.subject_to(x[:,0] == p)


    opti.solver('ipopt')
"""
def mpc_setup_dynamic(F, nx, nu, x0, N_horizon, l_end_mass, u_end_mass, min_DLI, max_DLI, u_min, u_max, data_dict, solver='ipopt'):
    # OPTIMAL CONTROL PROBLEM
    opti = ca.Opti()

    x = opti.variable(nx, N_horizon+1)
    u = opti.variable(nu, N_horizon)

    p = opti.parameter(nx, 1)
    energy = opti.parameter(1, N_horizon)
    # setting the objective
    obj = ca.dot(u[0,:], energy)
    
    # Parameters for penalties
    penalty_weight = 1e6  # Adjust this weight to increase/decrease the penalty

    # Penalty for being below lower bound
    penalty_below = ca.fmax(0, l_end_mass - x[2, -1])
    # Penalty for exceeding upper bound
    penalty_above = ca.fmax(0, x[2, -1] - u_end_mass)
    obj += penalty_weight * (penalty_below**2 + penalty_above**2)


    for k in range(N_horizon+1):
        if (k) % 24 == 0 and k > 0:
            penalty_below_DLI = ca.fmax(0, min_DLI - x[3,k] + x[3,k-24])
            penalty_above_DLI = ca.fmax(0,  x[3,k] - x[3,k-24] - max_DLI)
            obj += penalty_weight* (penalty_below_DLI**2 + penalty_above_DLI**2)
            
            
    opti.minimize(obj)


    for k in range(N_horizon):
        opti.subject_to(x[:,k+1] == F(x[:,k], u[:,k]))
        opti.subject_to([u[0,k] <= u_max, u[0,k] >= u_min])

    # Setting u_prev
    #opti.subject_to(u[1,0] == 0)
    for k in range(1, N_horizon):
        opti.subject_to(u[1, k] == u[0, k-1])

    opti.subject_to(x[:,0] == p)


    opti.solver('ipopt')

    return opti, obj, x, u,p,  energy


def mpc_setup_constant(F, nx, nu, x0, N_horizon, l_end_mass, u_end_mass, min_DLI, max_DLI, u_min, u_max, data_dict, solver='ipopt'):
    # OPTIMAL CONTROL PROBLEM
    opti = ca.Opti()

    x = opti.variable(nx, N_horizon+1)
    u = opti.variable(nu, N_horizon)

    p = opti.parameter(nx, 1)
    energy = opti.parameter(1, N_horizon)
    # setting the objective
    obj = ca.dot(u[0,:], energy)

    # Parameters for penalties
    penalty_weight = 1e6  # Adjust this weight to increase/decrease the penalty

    # Penalty for being below lower bound
    penalty_below = ca.fmax(0, l_end_mass - x[2, -1])
    # Penalty for exceeding upper bound
    penalty_above = ca.fmax(0, x[2, -1] - u_end_mass)
    obj += penalty_weight * (penalty_below**2 + penalty_above**2)


    for k in range(N_horizon+1):
        if (k) % 24 == 0 and k > 0:
            penalty_below_DLI = ca.fmax(0, min_DLI - x[3,k] + x[3,k-24])
            penalty_above_DLI = ca.fmax(0,  x[3,k] - x[3,k-24] - max_DLI)
            obj += penalty_weight* (penalty_below_DLI**2 + penalty_above_DLI**2)
            
            
    opti.minimize(obj)
    for k in range(N_horizon):
        opti.subject_to(x[:,k+1] == F(x[:,k], u[:,k]))
        opti.subject_to([u[0,k] <= u_max, u[0,k] >= 0])

    # Setting u_prev
    opti.subject_to(u[1,0] == 0)
    for k in range(1, N_horizon):
        opti.subject_to(u[0,k] == u[0,k-1])
        opti.subject_to(u[1, k] == u[0, k-1])

    opti.subject_to(x[:,0] == p)


    opti.solver('ipopt')

    return opti, obj, x, u,p,  energy



def get_min_max_DLI_points(TLI, min_DLI, max_DLI):

    last = 0

    min_points = []
    max_points = []
    for k in range(TLI.shape[0]):
        if (k) % 24 == 0 and k > 0:
            min_points.append((k, last + min_DLI))
            max_points.append((k, last + max_DLI))
            last = TLI[k]
    return min_points, max_points



def print_cost_and_energy_consumption(x, u, energy, total_energy, model=''):
    print('-'*20)
    print('Model type: ', model, ' End biomass: ', x[2,-1])
    print('Total cost: ')
    print(ca.dot(u[0,:].T, energy))
    print('Energy usage: ')
    print(total_energy)
    print('-'*20)



def get_params(opt_config):
    # Integrator settings
    N_horizon = opt_config['N_horizon']
    ts  = opt_config['ts']
    
    solver = opt_config['solver']

    # Growth settings
    photoperiod_length = opt_config['photoperiod_length']
    u_max = opt_config['u_max']
    u_min = opt_config['u_min']
    min_DLI = opt_config['min_DLI']
    max_DLI = opt_config['max_DLI']
    l_end_mass = opt_config['lower_terminal_mass']
    u_end_mass = opt_config['upper_terminal_mass']

    is_rate_degradation = opt_config['is_rate_degradation']
    c_degr = opt_config['c_degr']
    return N_horizon, ts, solver, photoperiod_length, u_max, u_min, min_DLI, max_DLI, l_end_mass, u_end_mass, is_rate_degradation, c_degr


def solve_mpc(opti, u_symbol, x_symbol, energy_values, min_DLI, max_DLI):
    sol = opti.solve()
    u_opt = sol.value(u_symbol)
    x_opt = sol.value(x_symbol)
    tot_cost_dynamic = ca.dot(u_opt[0,:].T, energy_values)
    tot_energy_dynamic = x_opt[3, -1].copy()
    # Get the coordinates for plotting the DLI boundaries
    min_points, max_points = get_min_max_DLI_points(TLI=x_opt[3,:], min_DLI=min_DLI, max_DLI=max_DLI)
    for l in range(0, x_opt.shape[1], 24):
        x_opt[3,l+1:] -= x_opt[3,l]
    return sol, u_opt, x_opt, tot_cost_dynamic, tot_energy_dynamic, min_points, max_points

def plot_photosynthesis(Crop, Env, fun_type='rectangular', block=True):
    def f_phot_max_values(PAR_flux_values):
        values = [Crop.return_photosynthesis(Env.CO2, Env.T_air, aerodynamical_resistance_eq(Env.air_vel,4, Crop.leaf_diameter), stomatal_resistance_eq(par), par, fun_type=fun_type) for par in PAR_flux_values]
        return values
    PAR_flux_values = np.linspace(0, 4000, 400)  # From 0 to 2000 umol/m^2/s with 400 points

    print('average from 200 + 200:')
    print(np.average(f_phot_max_values(np.array([200,200]))))
    baseline = np.average(f_phot_max_values(np.array([200,200])))
    print('average from 150 + 250:')
    print(np.average(f_phot_max_values(np.array([150,250])))/baseline)
    print('average from 100 + 300:')
    print(np.average(f_phot_max_values(np.array([100,300])))/baseline)
    print('average from 0 + 400:')
    print(np.average(f_phot_max_values(np.array([0,400])))/baseline)


    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(PAR_flux_values, f_phot_max_values(PAR_flux_values), label='f_phot_max vs PAR_flux')
    plt.title(f'Photosynthetic Efficiency as a Function of PAR Flux {fun_type}')
    plt.xlabel('PAR Flux (umol/m^2/s)')
    plt.ylabel('Photosynthetic Efficiency (f_phot_max)')
    plt.grid(True)
    plt.legend()
    plt.show(block=block)