import numpy as np
from typing import Dict, Any
from models.hvac import HVACModel
from models.utils import *
from config.utils import get_attribute
from models.utils import calculate_absolute_humidity

class ClimateModel:
    def __init__(self, config: Dict[str, Any]):
        self.hvac_model = HVACModel(config)

        # Get constants and parameters from config
        self.p = get_attribute(config, 'p')     
        self.rho_air = get_attribute(config, 'rho_air')  
        self.c_air = get_attribute(config, 'c_air')       
        self.c_omega = get_attribute(config, 'c_omega') 
        self.c_p = get_attribute(config, 'c_p')
        self.c_env = get_attribute(config, 'c_env')

        self.alpha_env = get_attribute(config, 'alpha_env')
        self.air_vel = get_attribute(config, 'air_vel')
        self.CO2_out = get_attribute(config, 'CO2_out')

        self.Lambda = get_attribute(config, 'lambda')    

        # Get simulation parameters
        self.A_env = get_attribute(config, 'A_env')
        self.A_crop = get_attribute(config, 'A_crop')
        self.V_in = get_attribute(config, 'V_in')
        self.V_hvac = get_attribute(config, 'V_hvac')

        self.C_in = self.rho_air*self.c_air*self.V_in    
        self.C_hvac = self.rho_air*self.c_air*self.V_hvac
        self.C_env = self.rho_air*self.c_env*self.V_in

        self.eta_light = get_attribute(config, 'eta_light')
        self.c_r = get_attribute(config, 'c_r')

        # Desired values
        self.T_desired = get_attribute(config, 'T_desired')
        self.RH_desired = get_attribute(config, 'RH_desired')
        self.CO2_desired = get_attribute(config, 'CO2_desired')

        self.Chi_desired = calculate_absolute_humidity(self.p, self.T_desired, self.RH_desired)

        # Initial state
        self.T_in = self.T_desired
        self.Chi_in = self.Chi_desired
        self.CO2_in = self.CO2_desired

        self.T_env = self.T_in
        self.T_sup = self.T_in
        self.Chi_sup = self.Chi_in
        self.T_crop = self.T_in

        self.init_state = np.array([self.T_in, self.Chi_in, self.CO2_in, self.T_env, self.T_sup, self.Chi_sup])

    def set_crop_model(self, crop_model):
        self.crop_model = crop_model

    def _update_state(self, T_in, Chi_in, CO2_in, T_env, T_sup, Chi_sup):
        self.T_in = T_in
        self.Chi_in = Chi_in
        self.CO2_in = CO2_in
        self.T_env = T_env
        self.T_sup = T_sup
        self.Chi_sup = Chi_sup
        self.T_crop = T_in

    def print_attributes(self, *args):
        if args:
            for attr_name in args:
                print(f"{attr_name}: {getattr(self, attr_name, 'Attribute not found')}")
        else:
            for attr, value in vars(self).items():
                print(f"{attr}: {value}")

    def temperature_ODE(self, U_par, Chi_crop, T_sup, u_sup, CAC, LAI, r_stm, r_bnd):
        Q_env = self.alpha_env*self.A_env*(self.T_env - self.T_in)

        Q_trans = LAI * self.Lambda * self.A_crop * (Chi_crop - self.Chi_in) / (r_stm + r_bnd)

        P_light = U_par/self.eta_light
        Q_light_ineff = (1-self.eta_light)*P_light
        R_net = (1-self.c_r)*U_par*CAC
        Q_sens_reflected = U_par - R_net

        Q_light = Q_light_ineff + Q_sens_reflected

        Q_hvac = u_sup*self.rho_air*self.c_air*(T_sup - self.T_in)

        return (1/self.C_in)*(Q_env + Q_trans + Q_light + Q_hvac)
    
    def env_temperature_ODE(self, T_out):
        return (1/self.C_env)*(self.alpha_env*self.A_env*(self.T_in - self.T_env) + self.alpha_env*self.A_env*(T_out - self.T_env))
    
    def sup_temperature_ODE(self, u_sup, T_hvac):
        return (1/self.C_hvac)*u_sup*self.rho_air*self.c_air*(T_hvac - self.T_sup)

    def humidity_ODE(self, Chi_sup, Chi_crop, u_sup, LAI, r_stm, r_bnd):
        Phi_trans = LAI * self.A_crop * (Chi_crop - self.Chi_in) / (r_stm + r_bnd)

        Phi_hvac = u_sup*(Chi_sup - self.Chi_in)

        return (1/self.V_in)*(Phi_trans + Phi_hvac)
    
    def sup_humidity_ODE(self, u_sup, Chi_hvac):
        return (1/self.V_hvac)*(u_sup*(Chi_hvac - self.Chi_sup))
    
    def CO2_ODE(self, f_phot, LAI, u_sup, Phi_c_inj):
        Phi_c_ass = f_phot*self.A_crop

        Phi_c_hvac = u_sup*self.c_omega*(self.CO2_out - self.CO2_in)

        return (1/(self.V_in*self.c_omega))*(Phi_c_inj - Phi_c_ass + Phi_c_hvac)

    def combined_ODE(self, state, control_inputs, data, hvac_input):
        T_in, Chi_in, CO2_in, T_env, T_sup, Chi_sup, X_ns, X_s = state
        self._update_state(T_in, Chi_in, CO2_in, T_env, T_sup, Chi_sup)
        
        LAI = self.crop_model.LAI
        CAC = self.crop_model.CAC
        f_phot = self.crop_model.f_phot

        u_rot, u_sup, u_cool, u_heat, u_humid, u_c_inj, PPFD = control_inputs
        T_out, RH_out, electricity_prices = data
        
        U_par = PPFD * self.c_p

        T_hvac, Chi_hvac, Chi_out = hvac_input
        self.__setattr__('Chi_hvac', Chi_hvac)
        self.__setattr__('T_hvac', T_hvac)
        self.__setattr__('Chi_out', Chi_out)
        
        r_stm = stomatal_resistance_eq(PPFD=PPFD)
        r_bnd = aerodynamical_resistance_eq(uninh_air_vel=self.air_vel, LAI=LAI, leaf_diameter=self.crop_model.leaf_diameter)
        Chi_crop = calculate_absolute_humidity(self.p, self.T_crop, 100)

        dT_in_dt = self.temperature_ODE(U_par, Chi_crop, T_sup, u_sup, CAC, LAI, r_stm, r_bnd)
        dChi_in_dt = self.humidity_ODE(Chi_sup, Chi_crop, u_sup, LAI, r_stm, r_bnd)
        dCO2_in_dt = self.CO2_ODE(f_phot, LAI, u_sup, u_c_inj)
        dT_env_dt = self.env_temperature_ODE(T_out)
        dT_sup_dt = self.sup_temperature_ODE(u_sup, T_hvac)
        dChi_sup_dt = self.sup_humidity_ODE(u_sup, Chi_hvac)
        
        return np.array([dT_in_dt, dChi_in_dt, dCO2_in_dt, dT_env_dt, dT_sup_dt, dChi_sup_dt])
