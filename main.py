from model import Model
from plot import Plotter
from scipy.integrate import solve_ivp
import numpy as np
from lights import PAR_generator, PPFD_generator
from config import *
from utilities import *


def main():
    plot = True
    model = Model()
    days = 25
    # Simulation parameters
    t_end = 60*60*24*days # one week (seconds)
    t_eval = np.linspace(0,t_end,endpoint=True,num=t_end)
    num_iter = 24*days # one iteration per hour
    samples_per_iter = int(t_end/num_iter)

    # Iteration parameters
    cur_t_eval = np.linspace(0,samples_per_iter,num=samples_per_iter) # time interval for current iteration
    cur_index_i = 0

    # Initial conditions
    env_init = np.array([info['init_value'] for info in env_states_info.values()])
    crop_init = np.array([info['init_value'] for info in crop_states_info.values()])
    y0 = np.concatenate((env_init, crop_init), axis=None)
    num_states = len(env_states_info) + len(crop_states_info)
    solutions = np.zeros([num_states, len(t_eval)], dtype=float)

    # Initialize plotter
    if plot:
        plotter = Plotter(t_eval, solutions)

    # Run simulation
    for i in range(num_iter):
        # (TODO): Fetch time-varying climate parameters
        
        outside_temperature = np.random.randint(20,26)
        outside_vapour_concentration = np.random.randint(45, 55)
        
        PAR_flux = PAR_generator(i)
        PPFD = PPFD_generator(i)
        wind_vel = 0.2
        DAT = i // 24 + 1
        print('Dat', DAT)
        climate = (outside_temperature, outside_vapour_concentration, DAT)
        control_input = (PAR_flux, PPFD, wind_vel)
        args = [climate, control_input]

        print('\n'.join([str(integer) for integer in climate]))
        print('-'*20)
        print('\n'.join([str(integer) for integer in control_input]))
        print('--')

        # Solve initial value problem for system of ODEs
        sol = solve_ivp(fun=model.model,
                        t_span=[cur_t_eval[0], cur_t_eval[-1]],
                        y0=y0,
                        method='RK45',
                        t_eval=cur_t_eval,
                        rtol=1e-5,
                        dense_output=True,
                        args=args)

        # Update iteration parameters
        cur_index_f = cur_index_i + len(sol.t)
        cur_t_eval = np.linspace(t_eval[cur_index_i], t_eval[cur_index_f], endpoint=True, num=samples_per_iter)

        y0 = sol.y[:,-1]
        X_ns = y0[-2]
        X_s = y0[-1]
        LAI = biomass_to_LAI(X_s)
        print('LAI: ', LAI)
        CAC = LAI_to_CAC(LAI)
        print('CAC: ', CAC)
        model.crop_model.print_attributes()
        solutions[:,cur_index_i:cur_index_f] = sol.y

        cur_index_i = cur_index_f - 1
        if plot:
            plotter.update_plot(t_eval, solutions, cur_index_f)

if __name__ == "__main__":
    main()
