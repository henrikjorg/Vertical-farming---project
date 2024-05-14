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
states_labels = ['X_s', 'X_ns', 'Fresh weight', 'DLI']
labels_to_plot =['Fresh weight', 'DLI']
Crop = CropModel(crop_config)
Env = EnvironmentModel(env_config)
x0 = ca.vertcat(Crop.X_ns, Crop.X_s, Crop.fresh_weight_shoot_per_plant, 0)
energy_values = data_dict['energy'][:N_horizon].copy()


plot_photosynthesis(Crop, Env, 'rectangular', block=False)

plot_photosynthesis(Crop, Env, 'exponential', block=False)
# --------------------------- Dynamic Model ------------------#
# Create a crop model
F, nx, nu= get_explicit_model(Crop, Env, is_rate_degradation=is_rate_degradation, c_degr=c_degr, ts=ts ,photoperiod_length=photoperiod_length, saturation_curve_type='rectangular')


# Set up the cost and constraints
opti, obj, x, u, p, energy = mpc_setup_dynamic(F=F, nx=nx, nu=nu, x0=x0, N_horizon=N_horizon, l_end_mass=l_end_mass, u_end_mass=u_end_mass,
                              min_DLI=min_DLI, max_DLI=max_DLI, u_min=u_min, u_max=u_max, data_dict=data_dict, solver=solver)
opti.set_value(p, x0)
opti.set_value(energy, energy_values)

#M = opti.to_function('M', [p, energy], [x, u], ['p', 'energy'], ['x_opt', 'u_opt'])
#[x_opt, u_opt] = M(x0, energy_values)

sol, u_opt, x_opt, tot_cost_dynamic, tot_energy_dynamic, min_points, max_points = solve_mpc(opti, u, x, energy_values, min_DLI, max_DLI)
#------------------------------------------------------------- #

# --------------------------- Dynamic Model with exponential photosynhesis curve ------------------#
# Create a crop model
F_exp, nx, nu= get_explicit_model(Crop, Env, is_rate_degradation=is_rate_degradation, c_degr=c_degr, ts=ts ,photoperiod_length=photoperiod_length, saturation_curve_type='exponential')


# Set up the cost and constraints
opti_exp, obj_exp, x_exp, u_exp, p_exp, energy_exp = mpc_setup_dynamic(F=F_exp, nx=nx, nu=nu, x0=x0, N_horizon=N_horizon, l_end_mass=l_end_mass, u_end_mass=u_end_mass,
                              min_DLI=min_DLI, max_DLI=max_DLI, u_min=u_min, u_max=u_max, data_dict=data_dict, solver=solver)
opti_exp.set_value(p_exp, x0)
opti_exp.set_value(energy_exp, energy_values)

#M = opti.to_function('M', [p, energy], [x, u], ['p', 'energy'], ['x_opt', 'u_opt'])
#[x_opt, u_opt] = M(x0, energy_values)

sol_exp, u_opt_exp, x_opt_exp, tot_cost_dynamic_exp, tot_energy_dynamic_exp, min_points_exp, max_points_exp = solve_mpc(opti_exp, u_exp, x_exp, energy_values, min_DLI, max_DLI)
#------------------------------------------------------------- #

# ------------------------ Constant model ------------------------- #
# Constant input

F_static, nx, nu = get_explicit_model(Crop, Env, is_rate_degradation=is_rate_degradation, c_degr=c_degr, ts=ts, photoperiod_length=photoperiod_length)

opti_const, obj_const, x_const, u_const, p_const, energy_const = mpc_setup_constant(F=F_static, nx=nx, nu=nu, x0=x0, N_horizon=N_horizon, l_end_mass=l_end_mass, u_end_mass=u_end_mass,
                              min_DLI=min_DLI, max_DLI=max_DLI, u_min=u_min, u_max=u_max, data_dict=data_dict, solver=solver)

opti_const.set_value(p_const, x0)
opti_const.set_value(energy_const, energy_values)

sol_const, u_opt_const, x_opt_const, tot_cost_const, tot_energy_const, min_points_const, max_points_const = solve_mpc(opti_const, u_const, x_const, energy_values, min_DLI, max_DLI)
#------------------------------------------------------------- #

# --------------------------- Dynamic Model with rate degradation ------------------#
# Create a crop model
F_dyn_deg, nx, nu= get_explicit_model(Crop, Env, is_rate_degradation=True, c_degr=c_degr, ts=ts ,photoperiod_length=photoperiod_length)


# Set up the cost and constraints
opti_dyn_deg, _, x, u, p, energy = mpc_setup_dynamic(F=F_dyn_deg, nx=nx, nu=nu, x0=x0, N_horizon=N_horizon, l_end_mass=l_end_mass, u_end_mass=u_end_mass,
                              min_DLI=min_DLI, max_DLI=max_DLI, u_min=u_min, u_max=u_max, data_dict=data_dict, solver=solver)
opti_dyn_deg.set_value(p, x0)
opti_dyn_deg.set_value(energy, energy_values)

_, u_opt_dyn_deg, x_opt_dyn_deg, tot_cost_dynamic_deg, tot_energy_dynamic_deg, min_points_deg, max_points_deg = solve_mpc(opti_dyn_deg, u, x, energy_values, min_DLI, max_DLI)
#------------------------------------------------------------- #



#Print dynamic
print_cost_and_energy_consumption( x=x_opt, u=u_opt, energy=energy_values, total_energy=tot_energy_dynamic, model='dynamic rectangular')
#Print dynamic with exponential saturation curve
print_cost_and_energy_consumption( x=x_opt_exp, u=u_opt_exp, energy=energy_values, total_energy=tot_energy_dynamic_exp, model='dynamic exponential')
#Print constant 
print_cost_and_energy_consumption( x=x_opt_const, u=u_opt_const, energy=energy_values, total_energy=tot_energy_const, model='constant input')
#Print dynamic with growth degradation from input rate
print_cost_and_energy_consumption( x=x_opt_const, u=u_opt_const, energy=energy_values, total_energy=tot_energy_dynamic_deg, model='dynamic with input rate effect')

#print(tot_cost_const)
print(tot_cost_dynamic)
saving_flex = 100*(1-float(tot_cost_dynamic)/float(tot_cost_const))
saving_degr = 100*(1-float(tot_cost_dynamic_deg)/float(tot_cost_const))
saving_flex_exp = 100*(1-float(tot_cost_dynamic_exp)/float(tot_cost_const))
print(f'Compared to the constant light, the dynamic model with rectangular hyperbola saturation curve saved {saving_flex:.1f} percent of the total energy cost.')
print(f'Compared to the constant light, the dynamic model with exponential saturation curve saved {saving_flex_exp:.1f} percent of the total energy cost.')
print(f'The dynamic model with growth degradation from changes in light saved:  {saving_degr:.1f}')
# Plot dynamic

plot_crop_casadi(t=np.linspace(0,Tf, N_horizon + 1), u_max=u_max, u_min = u_min, U=u_opt, X_true=x_opt, energy_price_array=energy_values,
          photoperiod_array=data_dict['photoperiod'][:N_horizon+1], eod_array=data_dict['eod'][:N_horizon], states_labels=states_labels, labels_to_plot=labels_to_plot, min_points = [], max_points=[],
          min_DLI=min_DLI, max_DLI=max_DLI,end_mass=l_end_mass, block=False, title=f'Dynamic rectangular - saved {saving_flex:.1f}% - total energy usage {tot_energy_dynamic:.1f}')

plot_crop_casadi(t=np.linspace(0,Tf, N_horizon + 1), u_max=u_max, u_min = u_min, U=u_opt_exp, X_true=x_opt_exp, energy_price_array=energy_values,
          photoperiod_array=data_dict['photoperiod'][:N_horizon+1], eod_array=data_dict['eod'][:N_horizon], states_labels=states_labels, labels_to_plot=labels_to_plot, min_points = [], max_points=[],
          min_DLI=min_DLI, max_DLI=max_DLI,end_mass=l_end_mass, block=False, title=f'Dynamic exponential - saved {saving_flex_exp:.1f}% - total energy usage {tot_energy_dynamic_exp:.1f}')


# Plot constant
plot_crop_casadi(t=np.linspace(0,Tf, N_horizon + 1), u_max=u_max, u_min = u_min, U=u_opt_const, X_true=x_opt_const, energy_price_array=energy_values,
          photoperiod_array=data_dict['photoperiod'][:N_horizon+1], eod_array=data_dict['eod'][:N_horizon], states_labels=states_labels,labels_to_plot=labels_to_plot, min_points = [], max_points=[],
          min_DLI=min_DLI, max_DLI=max_DLI,end_mass=l_end_mass, block=False, title=f'Constant input- total energy usage {tot_energy_const:.1f}')

# Plot dynamic with rate degradation
plot_crop_casadi(t=np.linspace(0,Tf, N_horizon + 1), u_max=u_max, u_min = u_min, U=u_opt_dyn_deg, X_true=x_opt_dyn_deg, energy_price_array=energy_values,
          photoperiod_array=data_dict['photoperiod'][:N_horizon+1], eod_array=data_dict['eod'][:N_horizon], states_labels=states_labels, labels_to_plot=labels_to_plot, min_points = [], max_points=[],
          min_DLI=min_DLI, max_DLI=max_DLI,end_mass=l_end_mass, block=True, title=f'Dynamic with rate degr - saved {saving_degr:.1f}% - total energy usage {tot_energy_dynamic_deg:.1f}')




"""
# ----------------------- Now try with uncertain energy prices --------------------#
energy_values_2 = energy_values
historical_volatility = np.std(energy_values)
energy_values_2[:24] = energy_values[:24]
for i in range(24, len(energy_values_2)):
    # Random noise added could be a fraction of the historical volatility
    random_noise = np.random.normal(0, historical_volatility * 0.1)
    energy_values_2[i] = energy_values[i] + random_noise
opti.set_value(p, x0)
opti.set_value(energy, energy_values_2)
sol, u_opt_noise, x_opt_noise, tot_cost_dynamic_noise, tot_energy_dynamic_noise, min_points_noise, max_points_noise = solve_mpc(opti, u, x, energy_values_2, min_DLI, max_DLI)
# ------------------------------------------------------------ #
"""