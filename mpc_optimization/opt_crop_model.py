# This file is for acados implementation on the SIMO system, where all environmental 
# parameters are constant except for light intensity

from acados_template import AcadosModel
from casadi import SX, vertcat, exp, sin, cos
from utilities import *
import numpy as np

def export_biomass_ode_model(Crop,Env, Ts, N_horizon, Days_horizon, light_percent):
    model_name = 'crop_model_simo'
    # X: State variables [X_ns, X_s]
    # U: Control input [PPFD]
    
    # Unpack state variables
    X_ns = SX.sym('X_ns')
    X_s = SX.sym('X_s')
    z_cumsum = SX.sym('z_cumsum')
    FW_plant = SX.sym('FW_plant')
    Tot_cost = SX.sym('Tot_cost')
    DLI  = SX.sym('DLI')

    
    
    # Unpack control input
    PPFD = SX.sym('PPFD')
    

    # Parameters
    energy_price = SX.sym('energy_price')
    photoperiod = SX.sym('photoperiod')

    x = vertcat(X_ns, X_s, FW_plant, DLI, Tot_cost, z_cumsum)
    u = vertcat(PPFD)
    z =SX.sym('z')

    X_ns_dot = SX.sym('X_ns_dot')
    X_s_dot = SX.sym('X_s_dot')
    z_cumsum_dot = SX.sym('z_cumsum_dot')
    FW_plant_dot = SX.sym('FW_plant_dot')
    Tot_cost_dot = SX.sym('Tot_cost_dot')
    DLI_dot  = SX.sym('DLI_dot')
    xdot = vertcat(X_ns_dot, X_s_dot, FW_plant_dot, DLI_dot, Tot_cost_dot, z_cumsum_dot)
    # Define constants (as provided or calibrated)
    T_air = Env.T_air
    CO2_air = Env.CO2
    PAR_flux = PPFD * 0.217
    # Calculate stomatal and aerodynamic conductances (reciprocals of resistances)
    LAI = SLA_to_LAI(SLA=Crop.SLA, c_tau=Crop.c_tau, leaf_to_shoot_ratio=Crop.leaf_to_shoot_ratio, X_s=X_s, X_ns=X_ns)
    g_stm = 1 / stomatal_resistance_eq(PPFD)
    g_bnd = 1 / aerodynamical_resistance_eq(Env.air_vel, LAI, Crop.leaf_diameter)
    # Dynamics equations adapted for CasADi
    g_car = Crop.c_car_1 * T_air**2 + Crop.c_car_2 * T_air + Crop.c_car_3
    g_CO2 = 1 / (1 / g_bnd + 1 / g_stm + 1 / g_car)
    Gamma = Crop.c_Gamma * Crop.c_q10_Gamma ** ((T_air - 20) / 10)
    
    epsilon_biomass = Crop.c_epsilon * (CO2_air - Gamma) / (CO2_air + 2 * Gamma)
    f_phot_max = (epsilon_biomass * PAR_flux * g_CO2 * Crop.c_w * (CO2_air - Gamma)) / (epsilon_biomass * PAR_flux + g_CO2 * Crop.c_w * (CO2_air - Gamma))
    f_phot = (1 - np.exp(-Crop.c_K * LAI)) * f_phot_max
    f_resp = (Crop.c_resp_sht * (1 - Crop.c_tau) * X_s + Crop.c_resp_rt * Crop.c_tau * X_s) * Crop.c_q10_resp ** ((T_air - 25) / 10)
    dw_to_fw = (1 - Crop.c_tau) / (Crop.dry_weight_fraction * Crop.plant_density)
    r_gr = Crop.c_gr_max * X_ns / (Crop.c_gamma * X_s + X_ns) * Crop.c_q10_gr ** ((T_air - 20) / 10)
    dX_ns = Crop.c_a * f_phot - r_gr * X_s - f_resp #- (1 - Crop.c_beta) / Crop.c_beta * r_gr * X_s
    dX_s = r_gr * X_s
    #dt = SX.sym('dt')  # dt = 1 for regular steps, dt = -z_daily_cumsum at the start of each day to reset

    f_expl = vertcat(dX_ns,                               # Non-structural dry weight per m^2
                     dX_s,                                # Structural dry weight per m^2
                     (dX_ns + dX_s) * dw_to_fw,           # Fresh weight of the shoot of one plant
                     (photoperiod-1)*(-1)*PPFD/(Ts*Days_horizon)-photoperiod*0.001*DLI,#PPFD/(Ts*N_horizon),                 # Average PPFD per day
                     PPFD*energy_price/(N_horizon * Ts),  # Average hourly cost of energy for the prediction horizon
                     PPFD/(Ts*N_horizon)* light_percent)  # The average PPFD during light period
    
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = vertcat(f_impl, z - PPFD*photoperiod)
    model.f_expl_expr =f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = vertcat(energy_price, photoperiod)
    model.z = z# PPFD * photoperiod
    model.name = model_name
    # Return state derivatives
    return model