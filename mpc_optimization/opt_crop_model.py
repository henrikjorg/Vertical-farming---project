# This file is for acados implementation on the SIMO system, where all environmental 
# parameters are constant except for light intensity

from acados_template import AcadosModel
from casadi import SX, vertcat, exp, sin, cos
from utilities import *
import numpy as np
from math import exp

def export_biomass_ode_model(Crop,Env, Ts, N_horizon, photoperiod_length, darkperiod_length, min_DLI):
    model_name = 'crop_model_simo'
    # X: State variables [X_ns, X_s]
    # U: Control input [PPFD]
    
    # Unpack state variables
    X_ns = SX.sym('X_ns')
    X_s = SX.sym('X_s')
    z_cumsum = SX.sym('z_cumsum')
    FW_plant = SX.sym('FW_plant')
    Tot_cost = SX.sym('Tot_cost')
    DLI_start_of_day  = SX.sym('DLI_start_of_day')
    DLI_lower = SX.sym('DLI_lower')
    u_last  = SX.sym('u_last')
    
    # Unpack control input
    PPFD = SX.sym('PPFD')
    

    # Parameters
    energy_price = SX.sym('energy_price')
    photoperiod = SX.sym('photoperiod')
    end_of_day = SX.sym('end_of_day')
    start_of_night = SX.sym('start_of_night')

    x = vertcat(X_ns, X_s, FW_plant, DLI_start_of_day, DLI_lower, Tot_cost, z_cumsum, u_last)
    u = vertcat(PPFD)
    z_dp_constraint =SX.sym('z_dp_constraint') # Ensuring no light during dark period
    z_min_DLI_constraint = SX.sym('z_min_DLI_constraint')
    z_DLI_end_of_day = SX.sym('z_DLI_end_of_day')
    u_last_dot = SX.sym('u_last_dot')             # Ensuring minimum DLI

    X_ns_dot = SX.sym('X_ns_dot')
    X_s_dot = SX.sym('X_s_dot')
    z_cumsum_dot = SX.sym('z_cumsum_dot')
    FW_plant_dot = SX.sym('FW_plant_dot')
    Tot_cost_dot = SX.sym('Tot_cost_dot')
    DLI_start_of_day_dot  = SX.sym('DLI_start_of_day_dot')
    DLI_lower_dot = SX.sym('DLI_lower_dot')
    
    xdot = vertcat(X_ns_dot, X_s_dot, FW_plant_dot, DLI_start_of_day_dot, DLI_lower_dot, Tot_cost_dot, z_cumsum_dot, u_last_dot)
    # Define constants (as provided or calibrated)
    T_air = Env.T_air
    CO2_air = Env.CO2
    delay = 0
    if delay:
      PAR_flux = z_cumsum * 0.217
    else:
      PAR_flux = PPFD * 0.217
    
    # Calculate stomatal and aerodynamic conductances (reciprocals of resistances)
    LAI = SLA_to_LAI(SLA=Crop.SLA, c_tau=Crop.c_tau, leaf_to_shoot_ratio=Crop.leaf_to_shoot_ratio, X_s=X_s, X_ns=X_ns)
    g_stm = 1 / stomatal_resistance_eq(PPFD)
    g_bnd = 1 / aerodynamical_resistance_eq(Env.air_vel, LAI, Crop.leaf_diameter)
    # Dynamics equations adapted for CasADi
    g_car = Crop.c_car_1 * T_air**2 + Crop.c_car_2 * T_air + Crop.c_car_3
    g_CO2 = 1 / (1 / g_bnd + 1 / g_stm + 1 / g_car)
    Gamma = Crop.c_Gamma * Crop.c_q10_Gamma ** ((T_air - 20) / 10)
    
    #ULM_degratation = exp(- (PPFD - last_u)**2/400**2)
    epsilon_biomass = Crop.c_epsilon * (CO2_air - Gamma) / (CO2_air + 2 * Gamma)
    f_phot_max = (epsilon_biomass * PAR_flux * g_CO2 * Crop.c_w * (CO2_air - Gamma)) / (epsilon_biomass * PAR_flux + g_CO2 * Crop.c_w * (CO2_air - Gamma))
    f_phot = (1 - np.exp(-Crop.c_K * LAI)) * f_phot_max #* ULM_degratation
    f_resp = (Crop.c_resp_sht * (1 - Crop.c_tau) * X_s + Crop.c_resp_rt * Crop.c_tau * X_s) * Crop.c_q10_resp ** ((T_air - 25) / 10)
    dw_to_fw = (1 - Crop.c_tau) / (Crop.dry_weight_fraction * Crop.plant_density)
    r_gr = Crop.c_gr_max * X_ns / (Crop.c_gamma * X_s + X_ns) * Crop.c_q10_gr ** ((T_air - 20) / 10)
    dX_ns = Crop.c_a * f_phot - r_gr * X_s - f_resp - (1 - Crop.c_beta) / Crop.c_beta * r_gr * X_s
    dX_s = r_gr * X_s
    #dX_ns *= UL
    #dt = SX.sym('dt')  # dt = 1 for regular steps, dt = -z_daily_cumsum at the start of each day to reset

    f_expl = vertcat(dX_ns,                               # Non-structural dry weight per m^2
                     dX_s,                                # Structural dry weight per m^2
                     (dX_ns + dX_s) * dw_to_fw,           # Fresh weight of the shoot of one plant
                     (photoperiod-1)*(-1)*PPFD/(Ts * photoperiod_length)-photoperiod*0.001*(DLI_start_of_day),#PPFD/(Ts*N_horizon),                 # Average PPFD per day
                     (end_of_day-1)*0.001*DLI_lower + end_of_day*(DLI_start_of_day-min_DLI)/Ts,
                     PPFD*energy_price/(N_horizon * Ts),  # Average hourly cost of energy for the prediction horizon
                     (PPFD-z_cumsum)*0.001,  # The average PPFD during light period
                     0)#(PPFD-u_last)*0.001)#(1-u_last)*0.001) # u_last_dot    
                     #)
    f_impl = xdot - f_expl

    model = AcadosModel()
    z = vertcat(z_dp_constraint, z_DLI_end_of_day, z_min_DLI_constraint) # The last z variable UL_123 is recently added
    z_constraints = vertcat(PPFD,
                          (DLI_start_of_day + PPFD/(photoperiod_length)),
                            ((DLI_start_of_day + PPFD/(photoperiod_length)) - min_DLI))
    h_end = vertcat(DLI_start_of_day,
                    (DLI_start_of_day-min_DLI))
    #z_expr = z - vertcat(PPFD*photoperiod,
    #                      (DLI_start_of_day + PPFD/(photoperiod_length))*(1-photoperiod),
    #                        ((DLI_start_of_day + PPFD/(photoperiod_length)) - end_of_day * min_DLI)*(1-photoperiod))#((DLI_start_of_day + PPFD/(photoperiod_length)) - end_of_day * min_DLI)*(1-photoperiod)) # The last "4" is recently added
    
    # ((DLI_start_of_day + PPFD/(photoperiod_length)) - end_of_day * min_DLI)*(1-photoperiod)
    print('------------')
    #exp(-0.25*((PPFD-u_last)/400)**2)
    model.f_impl_expr = vertcat(f_impl, z -z_constraints)
    model.f_expl_expr =f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = vertcat(energy_price, photoperiod, end_of_day, start_of_night)
    model.z = z
    model.con_h_expr = z
    model.con_h_expr_e = h_end
    model.con_h_expr_0 = z
    model.name = model_name
    # Return state derivatives
    return model