import sys
 
# setting path
sys.path.append('../VF-SIMULATION')
import casadi as ca
import numpy as np
from mpc_optimization.opt_utils import *
from utilities import *
from crop import CropModel
from environment import EnvironmentModel

def plot_response_to_light(crop, env, light_intensities):
    """
    Plots various model variables as a response to different input light intensities.

    Parameters:
    - crop: CropModel instance with crop-related parameters and equations.
    - env: EnvironmentModel instance with environment-related parameters.
    - light_intensities: array of light intensity values to simulate.
    """
    # Variables to store the results
    f_phot_values = []
    f_resp_values = []
    r_gr_values = []
    dX_ns_values = []
    dX_s_values = []
    
    # Constants
    T_air = env.T_air
    CO2_air = env.CO2
    x_ns = 5  # Example value
    x_s = 10    # Example value
    
    for u_light in light_intensities:
        PAR_flux = u_light * 0.217
        LAI = SLA_to_LAI(SLA=Crop.SLA, c_tau=Crop.c_tau, leaf_to_shoot_ratio=Crop.leaf_to_shoot_ratio, X_s=x_s, X_ns=x_ns)  # Adjust based on your CropModel methods
        g_stm = 1 / stomatal_resistance_eq(u_light)

        g_bnd = 1 / aerodynamical_resistance_eq(Env.air_vel, LAI=LAI, leaf_diameter=Crop.leaf_diameter)


        # Dynamics equations adapted for CasADi
        g_car = Crop.c_car_1 * T_air**2 + Crop.c_car_2 * T_air + Crop.c_car_3
        g_CO2 = 1 / (1 / g_bnd + 1 / g_stm + 1 / g_car)
        
        
        Gamma = Crop.c_Gamma * Crop.c_q10_Gamma ** ((T_air - 20) / 10)

        #ULM_degratation = exp(- (PPFD - last_u)**2/400**2)
        epsilon_biomass = Crop.c_epsilon * (CO2_air - Gamma) / (CO2_air + 2 * Gamma)
        
        f_phot_max = (epsilon_biomass * PAR_flux * g_CO2 * Crop.c_w * (CO2_air - Gamma)) / (epsilon_biomass * PAR_flux + g_CO2 * Crop.c_w * (CO2_air - Gamma))

        f_phot = (1 - np.exp(-Crop.c_K * LAI)) * f_phot_max 
        f_resp = (Crop.c_resp_sht * (1 - Crop.c_tau) * x_s + Crop.c_resp_rt * Crop.c_tau * x_s) * Crop.c_q10_resp ** ((T_air - 25) / 10)
        dw_to_fw = (1 - Crop.c_tau) / (Crop.dry_weight_fraction * Crop.plant_density)
        r_gr = Crop.c_gr_max * x_ns / (Crop.c_gamma * x_s + x_ns + 0.001) * Crop.c_q10_gr ** ((T_air - 20) / 10)
        dX_ns = Crop.c_a * f_phot - r_gr * x_s - f_resp - (1 - Crop.c_beta) / Crop.c_beta * r_gr * x_s
        dX_s = r_gr * x_s
        
        f_phot_values.append(f_phot)
        
        f_resp_values.append(f_resp)
        
        r_gr_values.append(r_gr)
        dX_s_values.append(dX_s)
        
        # Calculate dX_ns
        dX_ns_values.append(dX_ns)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(light_intensities, f_phot_values, label='f_phot')
    plt.plot(light_intensities, f_resp_values, label='f_resp')
    plt.plot(light_intensities, r_gr_values, label='r_gr')
    plt.plot(light_intensities, dX_ns_values, label='dX_ns')
    plt.plot(light_intensities, dX_s_values, label='dX_s')
    
    plt.xlabel('Light Intensity (u_light)')
    plt.ylabel('Model Variables')
    plt.title('Response of Model Variables to Light Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()

# Getting the crop and environment models
crop_config = load_config('crop_model_config.json')
env_config = load_config('env_model_config.json')
opt_config = load_config('opt_config.json')

# Integrator settings
N_horizon = 96
ts  = 3600
T_end = N_horizon * ts
Tf = T_end


# Growth settings
photoperiod_length = 24
darkperiod_length = 0
u_max = 450
u_min = 0
min_DLI = 350
max_DLI = 450
l_end_mass = 10
u_end_mass = 1100
dynamic_lighting = True
rate_degradation = True
c_degr = 400
data_dict = generate_data_dict(photoperiod_length=photoperiod_length, darkperiod_length=darkperiod_length, Nsim=N_horizon, shrink=False, N_horizon=N_horizon)
Crop = CropModel(crop_config)
Env = EnvironmentModel(env_config)
#plot_response_to_light(Crop, Env, light_intensities=np.linspace(0, 1000, 100))
# Creating the states
x_ns = ca.MX.sym('x_ns')
x_s = ca.MX.sym('x_s')
x_fw = ca.MX.sym('x_fw')
DLI = ca.MX.sym('DLI')
u_light = ca.MX.sym('u_light')
u_prev = ca.MX.sym('u_prev')
#u_dot = ca.MX.sym('u_dot')

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
if rate_degradation:
    f_phot_max = (epsilon_biomass * PAR_flux * g_CO2 * Crop.c_w * (CO2_air - Gamma)) / (epsilon_biomass * PAR_flux + g_CO2 * Crop.c_w * (CO2_air - Gamma))*ca.exp(-((u_light-u_prev)/c_degr)**2)
else:
    f_phot_max = (epsilon_biomass * PAR_flux * g_CO2 * Crop.c_w * (CO2_air - Gamma)) / (epsilon_biomass * PAR_flux + g_CO2 * Crop.c_w * (CO2_air - Gamma))

f_phot = (1 - np.exp(-Crop.c_K * LAI)) * f_phot_max #* ULM_degratation
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


x0 = ca.vertcat(Crop.X_ns, Crop.X_s, Crop.fresh_weight_shoot_per_plant, 0)

f_day = ca.Function('f', [x, u_control], [f_expl_day], ['x', 'u_control'], ['ode'])


intg_options = {
    'tf' : ts,
    'simplify' : True
    
}

dae_day = {
    'x': x,
    'p': u_control,
    'ode': f_day(x, u_control)
}

intg_day = ca.integrator('intg', 'rk', dae_day, intg_options)

res_day = intg_day(x0=x, p= u_control)

x_next_day = res_day['xf']

# Simplifying API to (x,u) -> (x_next)
# F simulates the function for one time step
F_day = ca.Function('F', [x,u_control], [x_next_day], ['x', 'u_control'], ['x_next'])
# sim is a simplified method to simulate the whole horizon
sim_day = F_day.mapaccum(N_horizon)





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
    opti.subject_to(x[:,k+1] == F_day(x[:,k], u[:,k]))
    opti.subject_to([u[0,k] <= u_max, u[0,k] >= 0])

# Setting u_prev
opti.subject_to(u[1,0] == 0)
for k in range(1, N_horizon):
    opti.subject_to(u[1, k] == u[0, k-1])

opti.subject_to(x[:,0] == p)


opti.solver('ipopt')
opti.set_value(p, x0)

opti.set_value(energy, data_dict['energy'][:N_horizon])
sol = opti.solve()
tli = sol.value(x)[3,:]
last = 0
min_points = []
max_points = []
for k in range(len(tli)):
    if (k) % 24 == 0 and k > 0:
        min_points.append((k, last + min_DLI))
        max_points.append((k, last + max_DLI))
        last = tli[k]
print('Total cost: ')
print(ca.dot(sol.value(u)[0,:], sol.value(energy)))
print('Energy usage: ')
print(sol.value(x)[3, -1])
plot_crop_casadi(t=np.linspace(0,Tf, N_horizon + 1), u_max=u_max, u_min = u_min, U=sol.value(u), X_true=sol.value(x), energy_price_array=data_dict['energy'][:N_horizon],
          photoperiod_array=data_dict['photoperiod'][:N_horizon+1], eod_array=data_dict['eod'][:N_horizon], states_labels=['ns', 's', 'fw', 'dli'], min_points = min_points, max_points=max_points,
          end_mass=l_end_mass)