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
from mpc_optimization.opt_utils import plot_crop, generate_energy_price, generate_photoperiod_values, print_ocp_setup_details
from ..data.utils import fetch_electricity_prices
# import acados.interfaces.acados_template as at
def load_config(file_path: str) -> dict:
    """Load configuration from a JSON file."""
    if file_path == 'opt_config.json':
        with open('mpc_optimization/' + file_path, 'r') as file:
            return json.load(file)
    else:
        with open(file_path, 'r') as file:
            return json.load(file)
def main():
    crop_config = load_config('crop_model_config.json')
    env_config = load_config('env_model_config.json')
    opt_config = load_config('opt_config.json')
    ocp_type = opt_config['ocp_type']
    photoperiod = opt_config['photoperiod']
    darkperiod = opt_config['darkperiod']
    closed_loop = opt_config['closed_loop']
    
    Fmax = opt_config['Fmax']   # Fmax: The maximum allowed input
    Fmin = opt_config['Fmin']   # Fmin: The minimum allowed input
    DaysSim = opt_config['DaysSim']
    Tsim = DaysSim * 24 * 3600   # Tsim: The total time of the whole simulation (closed loop)
    Ts = opt_config['Ts']       # Ts: the sampling time of one step
    N_horizon = opt_config['N_horizon'] # N_horizon: The number of steps within the horizon
    Days_horizon = N_horizon / 24
    Tf = N_horizon * Ts         # Tf: The total time of the horizon
    if closed_loop:
        Nsim = int(np.floor(Tsim/Ts))
        
        energy_prices, closed_loop_prices = fetch_electricity_prices('data/Spotprices_norway.csv', length=N_horizon, length_sim=Nsim)#generate_energy_price(N_horizon=N_horizon, Nsim=Nsim)
    else:
        Nsim = N_horizon
        energy_prices, _  = fetch_electricity_prices('spotprices_norway.csv',length = N_horizon)#generate_energy_price(N_horizon=N_horizon)
    photoperiod_values = generate_photoperiod_values(photoperiod=photoperiod, darkperiod=darkperiod, N_horizon=N_horizon)
    Crop = CropModel(crop_config)
    Env = EnvironmentModel(env_config)
    use_RTI = False
    x0 = np.array([Crop.X_ns, Crop.X_s, Crop.fresh_weight_shoot_per_plant, 0,0,0])

    ocp_solver, integrator, ocp = opt_setup(Crop=Crop, Env=Env, opt_config=opt_config, energy_prices=energy_prices, photoperiod_values=photoperiod_values, x0=x0
                                       , Fmax=Fmax, Fmin=Fmin, N_horizon=N_horizon, Days_horizon = Days_horizon,Ts=Ts, Tf=Tf, ocp_type=ocp_type,RTI=use_RTI)
    for i in range(N_horizon):
        ocp_solver.set(i, 'p', np.array([energy_prices[i], photoperiod_values[i]]))
        #ocp_solver.set(N_horizon+i, 'p', photoperiod_values[i])
    # Printing the OCP constraints and cost function (if available)
    print_ocp_setup_details(ocp)
    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu
    if not closed_loop: # Open loop
        
        simX = np.zeros((Nsim+1, nx))
        simU = np.zeros((Nsim, nu))
        simX[0,:] = x0

        status = ocp_solver.solve()
        ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

        if status != 0:
            raise Exception(f'acados returned status {status}.')

        # get solution
        for i in range(Nsim):
            simX[i,:] = ocp_solver.get(i, "x")
            simU[i,:] = ocp_solver.get(i, "u")
        simX[Nsim,:] = ocp_solver.get(Nsim, "x")
        print(ocp_solver.get_cost())
        plot_crop(np.linspace(0, Tf, Nsim+1), Fmax, Fmin, simU, simX, energy_price_array=energy_prices,latexify=False)
    else: # Closed loop

        Nsim = int(np.floor(Tsim/Ts))

        #raise ValueError()
        simX = np.zeros((Nsim+1, nx))
        simU = np.zeros((Nsim, nu))

        simX[0,:] = x0

        if use_RTI:
            t_preparation = np.zeros((Nsim))
            t_feedback = np.zeros((Nsim))

        else:
            t = np.zeros((Nsim))

        # do some initial iterations to start with a good initial guess
        num_iter_initial = 5
        for _ in range(num_iter_initial):
            ocp_solver.solve_for_x0(x0_bar = x0)

        # closed loop
        for i in range(Nsim):
            print(simX[i,:])
            print('*'*20)
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
                simU[i,:] = ocp_solver.solve_for_x0(x0_bar = simX[i, :])

                t[i] = ocp_solver.get_stats('time_tot')

            # simulate system
            simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])

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
        plot_crop(np.linspace(0, Tsim, Nsim+1), Fmax, Fmin, simU, simX,energy_price_array = closed_loop_prices ,latexify=False)
if __name__ == "__main__":
    main()
