#import sys
# setting path
#sys.path.append('../VF-SIMULATION')
import casadi as ca
import numpy as np
from opt_utils import *
from models.utils import *
from models.crop import CropModel
from models.climate import ClimateModel
from config.utils import load_config

# Getting the crop and environment models
config = load_config('../config/')
opt_config = load_opt_config('opt_config_casadi.json')

N_horizon, ts, solver, photoperiod_length, u_max, u_min, min_DLI, max_DLI, l_end_mass, u_end_mass, is_rate_degradation, c_degr = get_params(opt_config)
Tf = N_horizon * ts
data_dict = generate_data_dict(photoperiod_length=photoperiod_length, darkperiod_length=0, Nsim=N_horizon*20, shrink=False, N_horizon=N_horizon*20)
states_labels = ['X_s', 'X_ns', 'Fresh weight', 'DLI']
labels_to_plot =['Fresh weight', 'DLI']
Crop = CropModel(config)
Env = ClimateModel(config)
x0 = ca.vertcat(Crop.X_ns, Crop.X_s, Crop.fresh_weight_shoot_per_plant, 0)

F, nx, nu= get_explicit_model(Crop, Env, is_rate_degradation=is_rate_degradation, c_degr=c_degr, ts=ts ,photoperiod_length=photoperiod_length)


# Set up the cost and constraints
opti, obj, x, u, p, energy = mpc_setup_dynamic(F=F, nx=nx, nu=nu, x0=x0, N_horizon=N_horizon, l_end_mass=l_end_mass, u_end_mass=u_end_mass,
                              min_DLI=min_DLI, max_DLI=max_DLI, u_min=u_min, u_max=u_max, data_dict=data_dict, solver=solver)


apply_steps = 24

# Define the initial state
current_state = x0

# Initialize lists for storing simulation data
states = np.zeros((nx, N_horizon+1))

states[:,0] = current_state.T

control_actions = np.zeros((nu, N_horizon))


# Running the MPC in a closed loop

# Create array with moving average of last three days
num_days_moving_average = 10
ind_start = num_days_moving_average*apply_steps
true_energy_price = data_dict['energy'][ind_start:ind_start+N_horizon].copy()
for k in range(0, N_horizon, apply_steps):
    
    long_energy_price_array = data_dict['energy'][k:k+ind_start + N_horizon].copy()

    energy_price_with_moving_average = long_energy_price_array[ind_start:].copy()
    print(len(energy_price_with_moving_average))
    print(len(long_energy_price_array))
    
    
    for i in range(apply_steps, len(energy_price_with_moving_average)):
        print('i: ', i)
        print('---')
        print([[ind_start + i - k*apply_steps] for k in range(1, num_days_moving_average+1)])
        last_days_prices = [long_energy_price_array[ind_start + (i%24)-j*apply_steps] for j in range(1,num_days_moving_average+1)]
        energy_price_with_moving_average[i] = np.mean(last_days_prices)
    
    current_energy = energy_price_with_moving_average.copy()

    opti.set_value(p, x0)
    opti.set_value(energy, current_energy) 
    # Set the current state in the optimizer
    
    if k > 0:
        # Removing all constraints
        opti = update_opti(opti=opti, F=F, x=x, u=u, p=p,energy=energy, current_time=k, end_time=N_horizon-k, N_horizon=N_horizon,  l_end_mass=l_end_mass, u_end_mass=u_end_mass,
                     min_DLI=min_DLI, max_DLI=max_DLI, u_min=u_min, u_max=u_max, current_state=current_state, current_energy = current_energy)
        
    # Solve the MPC problem
    sol = opti.solve()
    
    # Extract the optimal control and state trajectories
    u_opt = sol.value(u)
    x_opt = sol.value(x)

    min_points, max_points = get_min_max_DLI_points(TLI=x_opt[3,:], min_DLI=min_DLI, max_DLI=max_DLI)
    min_points, max_points = get_min_max_DLI_points(TLI=np.zeros((1, x_opt[3,:].shape[0])), min_DLI=min_DLI, max_DLI=max_DLI)
    for l in range(0, N_horizon, apply_steps):
        x_opt[3,l+1:] -= x_opt[3,l]
    plot_crop_casadi(t=np.linspace(0,Tf, N_horizon + 1), u_max=u_max, u_min=u_min, U=u_opt, X_true=x_opt, energy_price_array=current_energy,
                 photoperiod_array=data_dict['photoperiod'][k:k+N_horizon+1], eod_array=data_dict['eod'][k:k+N_horizon], states_labels=states_labels, labels_to_plot=labels_to_plot, min_points = min_points, max_points=max_points,
          end_mass=l_end_mass, block=True,end_time=N_horizon-k, min_DLI=min_DLI, max_DLI=max_DLI, title='Dynamic closed loop')
    # Apply the first 24 control inputs (or the remaining if less than 24 hours left)
    u_apply = u_opt[:, :min(apply_steps, N_horizon - k)]
    
    # Simulate the system dynamics for each control input applied
    for j in range(u_apply.shape[1]):
        # Assuming F is a function that takes current_state and control input and returns next_state
        next_state = F(current_state, u_apply[:, j])
        states[:, k+j+1] = next_state.T
        control_actions[:,k + j] = u_apply[:, j]
        current_state = next_state  # Update current state


tot_cost_dynamic = ca.dot(control_actions[0,:], true_energy_price)
tot_energy_dynamic = x_opt[3, -1]
for i in range(0, N_horizon, apply_steps):
    states[3, i:] -= states[3,i]
min_points, max_points = get_min_max_DLI_points(TLI=np.zeros((1, states[3,:].shape[0])), min_DLI=min_DLI, max_DLI=max_DLI)

print_cost_and_energy_consumption(x=states, u=control_actions,energy=true_energy_price, model='Dynamic closed loop', total_energy=tot_energy_dynamic)
plot_crop_casadi(t=np.linspace(0,Tf, N_horizon + 1), u_max=u_max, u_min=u_min, U=control_actions, X_true=states, energy_price_array=true_energy_price,
                 photoperiod_array=data_dict['photoperiod'][:N_horizon+1], eod_array=data_dict['eod'][:N_horizon], states_labels=states_labels, labels_to_plot=labels_to_plot, min_points = min_points, max_points=max_points,
          end_mass=l_end_mass, block=True, min_DLI=min_DLI, max_DLI=max_DLI, title='Dynamic closed loop')