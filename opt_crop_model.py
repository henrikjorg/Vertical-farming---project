# This file is for acados implementation on the SIMO system, where all environmental 
# parameters are constant except for light intensity

from acados_template import AcadosModel
from casadi import SX, vertcat, exp
from utilities import *
import numpy as np
def export_biomass_ode_model(Crop, uninh_air_vel=0.2):
    model_name = 'crop_model_simo'
    # X: State variables [X_ns, X_s]
    # U: Control input [PPFD]
    
    # Unpack state variables
    X_ns = SX.sym('X_ns')
    X_s = SX.sym('X_s')
    
    x = vertcat(X_ns, X_s)
    # Unpack control input
    PPFD = SX.sym('PPFD')
    u = vertcat(PPFD)

    X_ns_dot = SX.sym('X_ns_dot')
    X_s_dot = SX.sym('X_s_dot')
    xdot = vertcat(X_ns_dot, X_s_dot)
    # Define constants (as provided or calibrated)
    T_air = 24
    CO2_air = 1200
    PAR_flux = PPFD * 0.217
    # Calculate stomatal and aerodynamic conductances (reciprocals of resistances)
    g_stm = 1 / stomatal_resistance_eq(PPFD)
    g_bnd = 1 / aerodynamical_resistance_eq(uninh_air_vel, Crop.LAI, Crop.leaf_diameter)
    
    # Dynamics equations adapted for CasADi
    g_car = Crop.c_car_1 * T_air**2 + Crop.c_car_2 * T_air + Crop.c_car_3
    g_CO2 = 1 / (1 / g_bnd + 1 / g_stm + 1 / g_car)
    Gamma = Crop.c_Gamma * Crop.c_q10_Gamma ** ((T_air - 20) / 10)
    
    epsilon_biomass = c_epsilon_calibrated(PPFD, T_air) * (CO2_air - Gamma) / (CO2_air + 2 * Gamma)
    f_phot_max = (epsilon_biomass * PAR_flux * g_CO2 * Crop.c_w * (CO2_air - Gamma)) / (epsilon_biomass * PAR_flux + g_CO2 * Crop.c_w * (CO2_air - Gamma))
    f_phot = (1 - exp(-Crop.c_K * Crop.LAI)) * f_phot_max
    f_resp = (Crop.c_resp_sht * (1 - Crop.c_tau) * X_s + Crop.c_resp_rt * Crop.c_tau * X_s) * Crop.c_q10_resp ** ((T_air - 25) / 10)
    
    r_gr = c_gr_max_calibrated(PPFD, T_air) * X_ns / (Crop.c_gamma * X_s + X_ns) * Crop.c_q10_gr ** ((T_air - 20) / 10)
    dX_ns = Crop.c_a * f_phot - r_gr * X_s - f_resp - (1 - c_beta_calibrated(PPFD, T_air)) / c_beta_calibrated(PPFD, T_air) * r_gr * X_s
    dX_s = r_gr * X_s
    f_expl = vertcat(dX_ns,
                     dX_s)
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.name = model_name

    # Return state derivatives
    return model