
import numpy as np
import matplotlib.pyplot as plt
from crop import CropModel
from environment import EnvironmentModel
from utilities import *
import json
from casadi import SX, vertcat, exp, sum1
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
from opt_crop_model import export_biomass_ode_model
from opt_setup_solver import opt_setup
from opt_utils import plot_crop, generate_energy_price, generate_photoperiod_values, print_ocp_setup_details
# import acados.interfaces.acados_template as at
def load_config(file_path: str) -> dict:
    """Load configuration from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def main():
    crop_config = load_config('crop_model_config.json')
    env_config = load_config('env_model_config.json')
    opt_config = load_config('opt_config.json')
    ocp_type = opt_config['ocp_type']
    N_horizon = opt_config['N_horizon']
    photoperiod = opt_config['photoperiod']
    darkperiod = opt_config['darkperiod']
    closed_loop = opt_config['closed_loop']
    energy_prices = generate_energy_price(N_horizon=N_horizon)
    photoperiod_values = generate_photoperiod_values(photoperiod=photoperiod, darkperiod=darkperiod, N_horizon=N_horizon)
    Crop = CropModel(crop_config)
    Env = EnvironmentModel(env_config)
    use_RTI = False
    x0 = np.array([Crop.X_ns, Crop.X_s, Crop.fresh_weight_shoot_per_plant, 0])
    Fmax = opt_config['Fmax']   # Fmax: The maximum allowed input
    Fmin = opt_config['Fmin']   # Fmin: The minimum allowed input
    DaysSim = opt_config['DaysSim']
    Tsim = DaysSim * 24 * 3600   # Tsim: The total time of the whole simulation (closed loop)
    Ts = opt_config['Ts']       # Ts: the sampling time of one step
    N_horizon = opt_config['N_horizon'] # N_horizon: The number of steps within the horizon
    Tf = N_horizon * Ts         # Tf: The total time of the horizon
    ocp_solver, integrator, ocp = opt_setup(Crop=Crop, Env=Env, opt_config=opt_config, energy_prices=energy_prices, photoperiod_values=photoperiod_values, x0=x0
                                       , Fmax=Fmax, Fmin=Fmin, N_horizon=N_horizon,Ts=Ts, Tf=Tf, ocp_type=ocp_type,RTI=use_RTI)
    for i in range(N_horizon):
        ocp_solver.set(i, 'p', energy_prices[i])
    # Printing the OCP constraints and cost function (if available)
    print_ocp_setup_details(ocp)
    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    if not closed_loop: # Open loop
        Nsim = N_horizon
        N = N_horizon
        simX = np.zeros((Nsim+1, nx))
        simU = np.zeros((Nsim, nu))

        simX[0,:] = x0
        simX = np.zeros((N+1, nx))
        simU = np.zeros((N, nu))

        status = ocp_solver.solve()
        ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

        if status != 0:
            raise Exception(f'acados returned status {status}.')

        # get solution
        for i in range(N):
            simX[i,:] = ocp_solver.get(i, "x")
            simU[i,:] = ocp_solver.get(i, "u")
        simX[N,:] = ocp_solver.get(N, "x")
        print(ocp_solver.get_cost())
        plot_crop(np.linspace(0, Tf, N+1), Fmax, Fmin, simU, simX, latexify=False)
    else: # Closed loop

        Nsim = int(np.floor(Tsim/Ts))
        N = Nsim
        print(N)
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
        plot_crop(np.linspace(0, Tsim, N+1), Fmax, Fmin, simU, simX, latexify=False)
if __name__ == "__main__":
    main()
