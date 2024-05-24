import numpy as np
from typing import Dict, Any
from models.hvac import HVACModel
from models.utils import *
from config.utils import get_attribute
from models.utils import calculate_absolute_humidity

class ClimateModel:
    def __init__(self, config: Dict[str, Any], cycle_duration_days):
        self.hvac_model = HVACModel(config)

        # Get constants and parameters from config
        self.p = get_attribute(config, 'p')     
        self.rho_air = get_attribute(config, 'rho_air')  
        self.c_air = get_attribute(config, 'c_air')       
        self.rho_c = get_attribute(config, 'rho_c') 
        self.c_p = get_attribute(config, 'c_p')
        self.C_hvac = get_attribute(config, 'C_hvac')
        self.c_duct = get_attribute(config, 'c_duct')
        self.rho_env = get_attribute(config, 'rho_env')
        self.c_env = get_attribute(config, 'c_env')
        self.x_env = get_attribute(config, 'x_env')

        self.alpha_env = get_attribute(config, 'alpha_env')
        self.alpha_ext = get_attribute(config, 'alpha_ext')
        self.air_vel = get_attribute(config, 'air_vel')
        self.CO2_out = get_attribute(config, 'CO2_out')

        self.Lambda = get_attribute(config, 'lambda')    

        # Get simulation parameters
        self.A_env = get_attribute(config, 'A_env')
        self.A_crop = get_attribute(config, 'A_crop')
        self.V_in = get_attribute(config, 'V_in')
        self.V_hvac = get_attribute(config, 'V_hvac')

        self.C_in = self.rho_air*self.c_air*self.V_in    
        self.C_env = self.rho_env*self.c_env*self.A_env*self.x_env

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

        # Initialize arrays to store intermediate data from solve_ivp function
        num_seconds = 24*60*60*cycle_duration_days + 1
        self.Q_data = np.zeros([4, num_seconds], dtype=float)
        self.Phi_data = np.zeros([2, num_seconds], dtype=float)
        self.Phi_c_data = np.zeros([3, num_seconds], dtype=float)

        self.t = 0

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

    def temperature_ODE(self, U_par, Chi_crop, u_sup, CAC, LAI, r_stm, r_bnd):
        Q_env = self.alpha_env*self.A_env*(self.T_env - self.T_in)

        P_light = (U_par*self.A_crop)/self.eta_light
        
        Q_ineff = (1-self.eta_light)*P_light
        Q_refl = U_par*CAC*self.c_r*self.A_crop
        Q_light = Q_ineff + Q_refl

        R_net = (U_par * CAC * (1 - self.c_r))*self.A_crop
        Q_sens_plant = (LAI * self.Lambda * self.A_crop * (Chi_crop - self.Chi_in) / (r_stm + r_bnd)) - R_net 

        # Q_hvac = u_sup*self.rho_air*self.c_air*(self.T_sup - self.T_in)
        # Q_hvac = -(Q_env + Q_trans + Q_light)
        Q_hvac = 0

        Qs = np.array([[Q_env, Q_sens_plant, Q_light, Q_hvac]]).T
        self.Q_data[:, self.t] = Qs[:, 0]
        
        # print("Q_env: ", Q_env)
        # print("Q sens plant: ", Q_sens_plant)
        # print("Q light: ", Q_light)
        # print("Q hvac: ", Q_hvac)
        # print()

        return (1/self.C_in)*(Q_env + Q_sens_plant + Q_light + Q_hvac)
    
    def env_temperature_ODE(self, T_out):
        return (1/self.C_env)*(self.alpha_env*self.A_env*(self.T_in - self.T_env) + self.alpha_ext*self.A_env*(T_out - self.T_env))
    
    def sup_temperature_ODE(self, u_sup, T_hvac):
        # return 0
        return (1/self.C_hvac)*u_sup*self.rho_air*self.c_duct*(T_hvac - self.T_sup)

    def humidity_ODE(self, Chi_crop, u_sup, LAI, r_stm, r_bnd):
        Phi_trans = LAI * self.A_crop * (Chi_crop - self.Chi_in) / (r_stm + r_bnd)

        # Phi_hvac = u_sup*(self.Chi_sup - self.Chi_in)
        Phi_hvac = - Phi_trans

        Phis = np.array([[Phi_trans, Phi_hvac]]).T
        self.Phi_data[:, self.t] = Phis[:, 0]

        # print("Phi_trans:", Phi_trans)
        # print("Phi_hvac:", Phi_hvac)
        # print()

        # return 0
        return (1/self.V_in)*(Phi_trans + Phi_hvac)
    
    def sup_humidity_ODE(self, u_sup, Chi_hvac):
        # return 0
        return (1/self.V_hvac)*(u_sup*(Chi_hvac - self.Chi_sup))
    
    def CO2_ODE(self, f_phot, u_sup, Phi_c_inj):
        Phi_c_ass = - f_phot*self.A_crop

        # Phi_c_hvac = u_sup*(self.rho_c/1000)*(self.CO2_out - self.CO2_in)
        Phi_c_hvac = 0

        Phi_c_inj = -(Phi_c_ass + Phi_c_hvac)

        Phis = np.array([[Phi_c_ass, Phi_c_hvac, Phi_c_inj]]).T
        self.Phi_c_data[:, self.t] = Phis[:, 0]

        return (1/(self.V_in*self.rho_c))*(Phi_c_inj + Phi_c_ass + Phi_c_hvac)

    def combined_ODE(self, t, current_step, state, control_inputs, data, hvac_input):
        self.t = int(t) + int(60*60*current_step)

        T_in, Chi_in, CO2_in, T_env, T_sup, Chi_sup, X_ns, X_s = state
        self._update_state(T_in, Chi_in, CO2_in, T_env, T_sup, Chi_sup)

        # TESTING WITH CONSTANT SUPPLY
        # self.T_sup = self.T_desired
        # self.Chi_sup = self.Chi_desired
        
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

        dT_in_dt = self.temperature_ODE(U_par, Chi_crop, u_sup, CAC, LAI, r_stm, r_bnd)
        dChi_in_dt = self.humidity_ODE(Chi_crop, u_sup, LAI, r_stm, r_bnd)
        dCO2_in_dt = self.CO2_ODE(f_phot, u_sup, u_c_inj)
        dT_env_dt = self.env_temperature_ODE(T_out)
        dT_sup_dt = self.sup_temperature_ODE(u_sup, T_hvac)
        dChi_sup_dt = self.sup_humidity_ODE(u_sup, Chi_hvac)
        
        return np.array([dT_in_dt, dChi_in_dt, dCO2_in_dt, dT_env_dt, dT_sup_dt, dChi_sup_dt])
