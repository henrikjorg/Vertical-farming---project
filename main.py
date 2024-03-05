from model import Model
from plot import Plotter
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from lights import PAR_generator, PPFD_generator
from config import *
from utilities import *


def main():
    plt.close('all')
    plot = True
    model = Model()
    days = 30
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
    crop_init = (model.crop_model.X_ns, model.crop_model.X_s) #np.array([info['init_value'] for info in crop_states_info.values()])
    y0 = np.concatenate((env_init, crop_init), axis=None)
    num_states = len(env_states_info) + len(crop_states_info)
    solutions = np.zeros([num_states, len(t_eval)], dtype=float)

    # Attributes to print
    crop_attributes_over_time = {
        'LAI': np.zeros(len(t_eval)),
        'dry_weight': np.zeros(len(t_eval)),
        'fresh_weight_shoot_per_plant': np.zeros(len(t_eval)),
        # Add any other attributes you want to track
    }


    # Initialize plotter
    if plot:
        plotter = Plotter(t_eval, solutions, crop_attributes_over_time)
    # Run simulation
    for i in range(num_iter):
        # (TODO): Fetch time-varying climate parameters
        
        outside_temperature = np.random.randint(20,26)
        outside_vapour_concentration = np.random.randint(45, 55)
        
        # Setting parameters
        PPFD = PPFD_generator(i)
        PAR_flux = PPFD * 0.217 # Conversion factor for the model because it was developed for solar radiation
        wind_vel = 0.2
        DAT = cur_index_i // (60*60*24)


        climate = (outside_temperature, outside_vapour_concentration, DAT)
        control_input = (PAR_flux, PPFD, wind_vel)
        args = [climate, control_input]

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

        # Update the internal attributes of the plant model
        model.crop_model.update_values(X_ns,X_s)
        crop_attributes_over_time['LAI'][cur_index_i:cur_index_f] = model.crop_model.LAI
        crop_attributes_over_time['dry_weight'][cur_index_i:cur_index_f] = model.crop_model.dry_weight
        crop_attributes_over_time['fresh_weight_shoot_per_plant'][cur_index_i:cur_index_f] = model.crop_model.fresh_weight_shoot_per_plant


        solutions[:,cur_index_i:cur_index_f] = sol.y
        cur_index_i = cur_index_f - 1
        if plot:
            plotter.update_plot(t_eval, solutions, cur_index_f, crop_attributes_over_time)

if __name__ == "__main__":
    main()
    plt.ioff()
    plt.show()
