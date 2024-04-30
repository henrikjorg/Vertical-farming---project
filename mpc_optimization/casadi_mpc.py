import sys
# setting path
sys.path.append('../VF-SIMULATION')
import casadi as ca
import numpy as np
from mpc_optimization.opt_utils import *
from utilities import *
from crop import CropModel
from environment import EnvironmentModel

# Getting the crop and environment models
crop_config = load_config('crop_model_config.json')
env_config = load_config('env_model_config.json')
opt_config = load_config('opt_config_casadi.json')

N_horizon, ts, solver, photoperiod_length, u_max, u_min, min_DLI, max_DLI, l_end_mass, u_end_mass, is_rate_degradation, c_degr = get_params(opt_config)
Tf = N_horizon * ts
data_dict = generate_data_dict(photoperiod_length=photoperiod_length, darkperiod_length=0, Nsim=N_horizon, shrink=False, N_horizon=N_horizon)

Crop = CropModel(crop_config)
Env = EnvironmentModel(env_config)
x0 = ca.vertcat(Crop.X_ns, Crop.X_s, Crop.fresh_weight_shoot_per_plant, 0)

# Create a crop model
F_day, nx, nu= get_explicit_model(Crop, Env, is_rate_degradation=is_rate_degradation, c_degr=c_degr, ts=ts ,photoperiod_length=photoperiod_length)

# Set up the cost and constraints
opti, obj, x, u, p, energy = mpc_setup_dynamic(F_day=F_day, nx=nx, nu=nu, x0=x0, N_horizon=N_horizon, l_end_mass=l_end_mass, u_end_mass=u_end_mass,
                              min_DLI=min_DLI, max_DLI=max_DLI, u_min=u_min, u_max=u_max, data_dict=data_dict, solver=solver)
# Doing some MPC
M = opti.to_function('M', [p, energy], [x, u], ['p', 'energy'], ['x_opt', 'u_opt'])
print('---')
print(M)
print('-----')
energy_values = data_dict['energy'][:N_horizon]
[x_opt, u_opt] = M(x0, energy_values)

X_log = []
U_log = []
print(u[0,:].shape)
print(energy_values.shape)
print(x_opt, u_opt)

# Solve the ocp
#sol = opti.solve()

# Get the coordinates for plotting the DLI boundaries
min_points, max_points = get_min_max_DLI_points(TLI=x_opt[3,:], min_DLI=min_DLI, max_DLI=max_DLI)

print_cost_and_energy_consumption( x=x_opt, u=u_opt, energy=energy_values)

# Plot the solution
plot_crop_casadi(t=np.linspace(0,Tf, N_horizon + 1), u_max=u_max, u_min = u_min, U=u_opt, X_true=x_opt, energy_price_array=data_dict['energy'][:N_horizon],
          photoperiod_array=data_dict['photoperiod'][:N_horizon+1], eod_array=data_dict['eod'][:N_horizon], states_labels=['ns', 's', 'fw', 'dli'], min_points = min_points, max_points=max_points,
          end_mass=l_end_mass)

