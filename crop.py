import numpy as np
from config import *
from utilities import *

# Modeling crop growth and plant transpiration.

class CropModel:
    def print_attributes(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")
    def biomass_ode(self, X_ns, X_s,T_air, CO2_air, PAR_flux, g_bnd, g_stm):
        CO2_ppm = CO2_air*100
        g_car = c_car_1*T_air**2 + c_car_2*T_air + c_car_3
        g_CO2 = 1/(1/g_bnd + 1/g_stm + 1/g_car)
        Gamma = c_Gamma*c_q10_Gamma**((T_air-20)/10)
        epsilon_biomass = c_epsilon * (CO2_ppm-Gamma)/(CO2_ppm + 2*Gamma)
        f_phot_max = (epsilon_biomass*PAR_flux*g_CO2*c_w*(CO2_ppm-Gamma))/(epsilon_biomass*PAR_flux+g_CO2*c_w*(CO2_ppm-Gamma))
        f_phot = (1-np.exp(-c_K*c_lar*(1-c_tau)*X_s))*f_phot_max
        f_resp = (c_resp_sht*(1-c_tau)*X_s+c_resp_rt*c_tau*X_s)*c_q10_resp**((T_air-25)/10)

        r_gr = c_gr_max*X_ns/(c_gamma*X_s+X_ns)*c_q10_gr**((T_air-20)/10)
        
        dX_ns = c_a*f_phot - r_gr*X_s - f_resp - (1-c_beta)/c_beta*r_gr*X_s

        dX_s = r_gr*X_s

        self.f_phot_max = f_phot_max
        self.f_phot = f_phot
        self.fraction_of_max_phot = f_phot/f_phot_max
        self.f_resp = f_resp
        self.r_gr = r_gr
        return dX_ns, dX_s
    def transpiration_ode(self):
        return 0

    def crop_conditions(self, env_state, crop_state, climate, control_input):
        T_air, Chi_air, CO2_air = env_state
        X_ns, X_s = crop_state
        LAI = biomass_to_LAI(X_s)

        # Varying parameters
        T_out, Chi_out, DAT = climate
        PAR_flux, PPFD ,wind_vel = control_input
        g_bnd = 1/aerodynamical_resistance_eq(wind_vel, LAI)
        g_stm = 1/stomatal_resistance_eq(PPFD)
        dNS_dt, dS_dt = self.biomass_ode(X_ns, X_s, T_air, CO2_air, PAR_flux, g_bnd, g_stm)
        return np.array([dNS_dt, dS_dt])