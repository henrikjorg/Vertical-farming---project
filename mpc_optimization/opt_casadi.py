import sys
 
# setting path
sys.path.append('../VF-SIMULATION')

import numpy as np
import matplotlib.pyplot as plt
from crop import CropModel
from environment import EnvironmentModel
from utilities import *
import json
from casadi import SX, vertcat, exp, sum1
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
from mpc_optimization.opt_crop_model import export_biomass_ode_model
from mpc_optimization.opt_setup_solver import opt_setup
from mpc_optimization.opt_utils import plot_crop, generate_energy_price, generate_photoperiod_values, generate_start_of_night_array, print_ocp_setup_details, generate_end_of_day_array
from data.utils import fetch_electricity_prices
INF = 1e12
# import acados.interfaces.acados_template as at
def load_config(file_path: str) -> dict:
    """Load configuration from a JSON file."""
    if file_path == 'opt_config.json':
        with open('mpc_optimization/' + file_path, 'r') as file:
            return json.load(file)
    else:
        with open(file_path, 'r') as file:
            return json.load(file)
def solver_set_new_horizon(solver, new_horizon):
    solver.solver_options["sim_method_num_stages"] = solver.solver_options["sim_method_num_stages"][:new_horizon]
    solver.solver_options["sim_method_jac_reuse"] = solver.solver_options["sim_method_jac_reuse"][:new_horizon]
    solver.solver_options["shooting_nodes"] = solver.solver_options["shooting_nodes"][:new_horizon]
    solver.solver_options["tf"] = solver.solver_options["shooting_nodes"][-1]
    solver.solver_options["time_steps"] = solver.solver_options["time_steps"][:new_horizon]
    solver.solver_options["sim_method_num_steps"] = solver.solver_options["sim_method_num_steps"][:new_horizon]
    solver.N = new_horizon
def update_parameters_and_constraints(N_horizon, data_dict, solver, integrator, max_DLI, sim_iter):
    for j in range(N_horizon):
        solver.set(j, 'p', np.array([data_dict['energy'][sim_iter + j], data_dict['photoperiod'][sim_iter + j], data_dict['eod'][sim_iter + j], data_dict['son'][sim_iter +j]]))
        if data_dict['eod'][sim_iter + j]:
                solver.constraints_set(j, "lh", np.array([-INF,-INF,0]))
                solver.constraints_set(j, "uh", np.array([INF, max_DLI, INF]))
        elif data_dict['photoperiod'][sim_iter + j]:
                solver.constraints_set(j, "lh", np.array([-1e-4,-INF,-INF]))
                solver.constraints_set(j, "uh", np.array([1e-4, INF, INF]))
    if data_dict['son'][sim_iter + N_horizon]:
        solver.constraints_set(N_horizon, "lh", np.array([-INF, -INF]))
        solver.constraints_set(N_horizon, "uh", np.array([max_DLI, INF]))

    integrator.set('p', np.array([data_dict['energy'][sim_iter], data_dict['photoperiod'][sim_iter], data_dict['eod'][sim_iter], data_dict['son'][sim_iter]]))

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

def main():
    crop_config = load_config('crop_model_config.json')
    env_config = load_config('env_model_config.json')
    opt_config = load_config('opt_config.json')
    ocp_type = opt_config['ocp_type']
    photoperiod_length = opt_config['photoperiod_length']
    darkperiod_length = opt_config['darkperiod_length']
    closed_loop = opt_config['closed_loop']
    states_labels = opt_config['states_labels']
    first_plot = opt_config['first_plot']
    Z_labels = opt_config['Z_labels']
    plot_Z = opt_config['plot_Z']
    if not plot_Z:
        Z_labels = None
    plot_every_stage = opt_config['plot_every_stage']
    shrink = opt_config['shrink_horizon']
    Fmax = opt_config['Fmax']   # Fmax: The maximum allowed input
    Fmin = opt_config['Fmin']   # Fmin: The minimum allowed input
    min_DLI = opt_config['min_DLI']
    max_DLI = opt_config['max_DLI']
    DaysSim = opt_config['DaysSim']
    Tsim = DaysSim * 24 * 3600   # Tsim: The total time of the whole simulation (closed loop)
    Ts = opt_config['Ts']       # Ts: the sampling time of one step
    N_horizon = opt_config['N_horizon'] # N_horizon: The number of steps within the horizon

    Tf = N_horizon * Ts         # Tf: The total time of the horizon
    if closed_loop:
        Nsim = int(np.floor(Tsim/Ts))
    else:
        Nsim = N_horizon
    data_dict = generate_data_dict(photoperiod_length=photoperiod_length, darkperiod_length=darkperiod_length, Nsim=Nsim, shrink=shrink, N_horizon=N_horizon)
    Crop = CropModel(crop_config)
    Env = EnvironmentModel(env_config)
    use_RTI = False
    x0 = np.array([Crop.X_ns, Crop.X_s, Crop.fresh_weight_shoot_per_plant, 0,0,0,0, 0])

    ocp_solver, integrator, ocp = opt_setup(Crop=Crop, Env=Env, opt_config=opt_config, x0=x0
                                       , Fmax=Fmax, Fmin=Fmin, N_horizon=N_horizon, Ts=Ts, Tf=Tf, ocp_type=ocp_type,RTI=use_RTI)

    update_parameters_and_constraints(N_horizon=N_horizon, data_dict=data_dict, solver=ocp_solver, integrator=integrator, max_DLI=max_DLI, sim_iter=0)
    
    # Printing the OCP constraints and cost function (if available)

    print_ocp_setup_details(ocp)
    
    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu
    nz = ocp_solver.acados_ocp.dims.nz
    simX = np.zeros((Nsim+1, nx))
    simU = np.zeros((Nsim, nu))
    simZ = np.zeros((Nsim, nz))
    simX[0,:] = x0
    # do some initial iterations to start with a good initial guess
    num_iter_initial =5
    for _ in range(num_iter_initial):
        ocp_solver.solve_for_x0(x0_bar = x0)
    if not closed_loop: # Open loop

        status = ocp_solver.solve()
        ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")
        if status != 0:
            raise Exception(f'acados returned status {status}.')

        # get solution
        for i in range(Nsim):
            simX[i,:] = ocp_solver.get(i, "x")
            simU[i,:] = ocp_solver.get(i, "u")
            simZ[i,:] = ocp_solver.get(i, "z")
        simX[Nsim,:] = ocp_solver.get(Nsim, "x")

        print(ocp_solver.get_cost())

        plot_crop(np.linspace(0, Tf, Nsim+1), Fmax, Fmin, simU, simX, energy_price_array=data_dict['energy'][:Nsim], photoperiod_array=data_dict['photoperiod'][:Nsim+1], Z_true= simZ, Z_labels=Z_labels,
                min_DLI= min_DLI, max_DLI=max_DLI,plot_all=True,
                  states_labels=states_labels,first_plot=first_plot, latexify=False)
    else: # Closed loop
        if Nsim < N_horizon:
            raise ValueError('The simulation length is initially set to be shorter than the horizon')
        
        if use_RTI:
            t_preparation = np.zeros((Nsim))
            t_feedback = np.zeros((Nsim))
        else:
            t = np.zeros((Nsim))

        
        
        # closed loop
        for i in range(Nsim):
            print('------')
            print('iter: ', i)
            
            if N_horizon <= Nsim-i or not shrink:
                N_shrinking = N_horizon
            else:
                N_shrinking = Nsim - i
                print('Horizon shrinked to: ', N_shrinking)
                solver_set_new_horizon(ocp_solver, N_shrinking)
                
            print('-------')
            # Only for testing purposes
            simX_temp = np.zeros((N_shrinking+1, nx))
            simX_temp2 = np.zeros((N_shrinking+1, nx))
            simZ_temp = np.zeros((N_shrinking, nz))
            simU_temp = np.zeros((N_shrinking, nu))
            if i > 0:
                update_parameters_and_constraints(N_horizon=N_shrinking, data_dict=data_dict, solver=ocp_solver, integrator=integrator, max_DLI=max_DLI, sim_iter=i)
                for _ in range(num_iter_initial):
                    ocp_solver.solve_for_x0(x0_bar = simX[i,:])
            # Testing stuff
            simX_temp[0,:] = simX[i,:]
            simX_temp2[0,:] = simX[i,:]

            if use_RTI:
                # preparation phase
                ocp_solver.options_set('rti_phase', 1)
                status = ocp_solver.solve()
                t_preparation[i] = ocp_solver.get_stats('time_tot')

                # set initial state
                ocp_solver.set(0, "lbx", simX[i, :])
                ocp_solver.set(0, "ubx", simX[i, :])

                # feedback phase
                ocp_solver.options_set('rti_phase', 2)
                status = ocp_solver.solve()
                t_feedback[i] = ocp_solver.get_stats('time_tot')

                simU[i, :] = ocp_solver.get(0, "u")
                
                
            else:
                # solve ocp and get next control input
                ocp_solver.solve_for_x0(x0_bar = simX[i,:])
                simU[i,:] = ocp_solver.solve_for_x0(x0_bar = simX[i, :])
                t[i] = ocp_solver.get_stats('time_tot')

            # simulate system
            simZ[i,:] = ocp_solver.get(0, "z")
            simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])

            # Testing the ocp solver and comparing it to the integrator
            if plot_every_stage:
                for j in range(N_shrinking):
                    simX_temp[j,:] = ocp_solver.get(j, "x")
                    simU_temp[j,:] = ocp_solver.get(j, "u")
                    simZ_temp[j,:] = ocp_solver.get(j, "z")
                simX_temp[N_shrinking,:] = ocp_solver.get(N_shrinking, "x")
                print(ocp_solver.get_cost())
                print(simX_temp[:,2])
                print(simU_temp)
                
                plot_crop(np.linspace(0, N_shrinking*Ts, N_shrinking+1), Fmax, Fmin, simU_temp, simX_temp, energy_price_array=data_dict['energy'][i:i + N_shrinking], photoperiod_array=data_dict['photoperiod'][i:i+N_shrinking],
                        Z_labels=Z_labels,Z_true=simZ_temp,min_DLI= min_DLI, max_DLI=max_DLI,plot_all=True,
                        states_labels=states_labels,first_plot=first_plot, latexify=False)
                # Testing to see the differences between integrator and solver
                for j in range(1,N_shrinking):
                   simX_temp2[j,:] = integrator.simulate(x=simX_temp2[j-1, :], u=simU_temp[j-1,:])
                simX_temp2[N_shrinking, :] = integrator.simulate(x=simX_temp2[-1, :], u=simU_temp[-1, :])
                print('-'*30)
                print('Simulated FW state from the OcpSolver')
                print(simX_temp[:,2])
                print('-'*40)
                print('Simulated FW state from the SimSolver')
                print(simX_temp2[:,2])


        # evaluate timings
        if use_RTI:
            # scale to milliseconds
            t_preparation *= 1000
            t_feedback *= 1000
            print(f'Computation time in preparation phase in ms: \
                    min {np.min(t_preparation):.3f} median {np.median(t_preparation):.3f} max {np.max(t_preparation):.3f}')
            print(f'Computation time in feedback phase in ms:    \
                    min {np.min(t_feedback):.3f} median {np.median(t_feedback):.3f} max {np.max(t_feedback):.3f}')
        else:
            # scale to milliseconds
            t *= 1000
            print(f'Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}')
        plot_crop(np.linspace(0, Tsim, Nsim+1), Fmax, Fmin, simU, simX,energy_price_array = data_dict['energy'][:Nsim], photoperiod_array=data_dict['photoperiod'][:Nsim],
                  Z_true=simZ, Z_labels=Z_labels,  min_DLI=min_DLI, max_DLI=max_DLI,
                  plot_all=True, states_labels=states_labels, first_plot=first_plot ,latexify=False)
if __name__ == "__main__":
    main()
