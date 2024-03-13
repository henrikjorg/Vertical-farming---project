from model import Model
from plot import Plotter
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from lights import PAR_generator, PPFD_generator
import time
import json

# Run parameters
days = 25
plot = True
eval_every_x_seconds = 60
env_attributes_to_plot = ['transpiration']
crop_attributes_to_plot = ['LAI', 'CAC', 'dry_weight', 'fresh_weight_shoot_per_plant', 'f_phot_converted']


def load_config(file_path: str) -> dict:
    """Load configuration from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def initialize_simulation(crop_config: dict, env_config: dict, days: int, eval_every_x_seconds: int) -> dict:
    """Set up simulation parameters and initial conditions."""
    # Simulation time parameters
    t_end = 60 * 60 * 24 * days  # Total simulation time in seconds
    num_iter = 24 * days  # One iteration per hour
    seconds_per_iter = int(t_end / num_iter)
    samples_per_iter = int(seconds_per_iter / eval_every_x_seconds)
    
    # Time evaluation points
    t_eval = np.linspace(0, t_end, endpoint=True, num=int(days * 24 * 60 * 60 / eval_every_x_seconds + 1))

    # Model and initial conditions
    model = Model(crop_config=crop_config, env_config=env_config)
    env_init = np.array([value for key, value in env_config.items() if key.startswith('init_')])
    crop_init = (model.crop_model.X_ns, model.crop_model.X_s)
    y0 = np.concatenate((env_init, crop_init), axis=None)
    
    return {
        "model": model, "t_eval": t_eval, "y0": y0, "samples_per_iter": samples_per_iter,
        "seconds_per_iter": seconds_per_iter, "num_iter": num_iter
    }

def run_simulation(sim_params: dict, plot: bool = True):
    """Execute the simulation loop."""
    # Unpack simulation parameters
    model, t_eval, y0, samples_per_iter, seconds_per_iter, num_iter = \
        [sim_params[key] for key in ("model", "t_eval", "y0", "samples_per_iter", "seconds_per_iter", "num_iter")]
    
    solutions = np.zeros((len(y0), len(t_eval)), dtype=float)  # Store solutions
    
    # Attributes to plot (these are just examples)
    crop_attributes_over_time = {}
    env_attributes_over_time = {}
    for attr in crop_attributes_to_plot:
        crop_attributes_over_time[attr] = np.zeros(len(t_eval))
    for attr in env_attributes_to_plot:
        env_attributes_over_time[attr] = np.zeros(len(t_eval))
    
    # Initialize Plotter
    plotter = Plotter(t_eval, solutions, crop_attributes_over_time, env_attributes_over_time) if plot else None

    cur_index_i = 0
    for i in range(num_iter):
        # Simulate external conditions
        hour = (i * seconds_per_iter) // (60 * 60)
        climate, control_input = simulate_external_conditions(hour)

        # Update simulation parameters
        sol = solve_ivp(fun=model.model, t_span=[0, seconds_per_iter], y0=y0, method='RK45', t_eval=np.linspace(0, seconds_per_iter, samples_per_iter),
                        args=[climate, control_input])

        # Process solution
        process_solution(sol, model, crop_attributes_over_time, env_attributes_over_time, cur_index_i, solutions)
        cur_index_i += len(sol.t)  # Update current index for next iteration
        y0 = sol.y[:, -1]  # Update initial conditions for next iteration
        
        if plot:
            plotter.update_plot(t_eval, solutions, t_eval[cur_index_i], crop_attributes_over_time, env_attributes_over_time)
    #if plot:
    #    plotter.plot_cac_vs_transpiration(crop_attributes_over_time, env_attributes_over_time)


def simulate_external_conditions(hour: int) -> tuple:
    """Simulate external climate and control input for the current simulation step."""
    outside_temperature = np.random.randint(20, 26)
    outside_vapour_concentration = np.random.randint(45, 55)
    PPFD = PPFD_generator(hour)
    PAR_flux = PPFD * 0.217  # Conversion factor for the model
    wind_vel = 0.2
    DAT = hour // 24
    
    climate = (outside_temperature, outside_vapour_concentration, DAT)
    control_input = (PAR_flux, PPFD, wind_vel)
    
    return climate, control_input

def process_solution(sol, model, crop_attrs, env_attrs, cur_index, solutions):
    """Update attributes and solutions based on the current simulation step."""
    # Extract relevant data from solution
    cur_index_f = cur_index + len(sol.t)
    solutions[:, cur_index:cur_index_f] = sol.y
    
    # Update tracked attributes
    for key in crop_attrs.keys():
        crop_attrs[key][cur_index:cur_index_f] = getattr(model.crop_model, key)
    for key in env_attrs.keys():
        env_attrs[key][cur_index:cur_index_f] = getattr(model.env_model, key)

def main():
    plt.close('all')
    crop_config = load_config('crop_model_config.json')
    env_config = load_config('env_model_config.json')
    sim_params = initialize_simulation(crop_config, env_config, days=days, eval_every_x_seconds=eval_every_x_seconds)
    run_simulation(sim_params, plot=plot)

    print(f"Execution time: {time.time() - start_time} seconds")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    main()
    # Optional: cProfile.run('main()', sort='cumtime')
