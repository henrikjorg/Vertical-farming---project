import numpy as np
from config import *
from utilities import *

# Modeling crop growth and plant transpiration.

class CropModel:
    def __init__(self, ideal_temp=24, ideal_PPFD=400, init_FW_per_plant=1, structural_to_nonstructural = 0.75, SLA = 300, dry_weight_fraction = 0.05, leaf_to_shoot_ratio = 0.92, plant_density=25):
        self.ideal_temp = ideal_temp
        self.ideal_PPFD = ideal_PPFD
        self.iter = 0
        self.DAT = 0
        self.plant_density = plant_density # Carotti had 25
        self.structural_to_nonstructural = structural_to_nonstructural
        self.dry_weight_fraction = dry_weight_fraction
        self.fresh_weight_shoot_per_plant = init_FW_per_plant
        self.fresh_weight_shoot = self.fresh_weight_shoot_per_plant*self.plant_density
        
        self.dry_weight = self.fresh_weight_shoot*self.dry_weight_fraction/(1-c_tau) 
        self.X_ns = self.dry_weight*(1-self.structural_to_nonstructural)
        self.X_s = self.dry_weight*self.structural_to_nonstructural
        self.fresh_weight_shoot_per_plant = self.fresh_weight_shoot/self.plant_density
        self.SLA = SLA
        self.leaf_to_shoot_ratio = leaf_to_shoot_ratio
        self.LAI = SLA_to_LAI(self.SLA, c_tau, self.leaf_to_shoot_ratio, self.X_s, self.X_ns)
        self.CAC = LAI_to_CAC(self.LAI)
    def set_SLA(self, temp=24, PPFD=400):
        # TODO
        return
    def set_fresh_weight_shoot(self, k=c_tau):
        # k is c_tau, or the ratio of root dry weight to the total dry weight for lettuce grown in soil
        self.fresh_weight_shoot = self.dry_weight*(1-k)/self.dry_weight_fraction
        self.fresh_weight_shoot_per_plant = self.fresh_weight_shoot/self.plant_density
    def update_values(self, X_ns,X_s):
        self.dry_weight = X_ns+X_s
        self.set_fresh_weight_shoot()
        self.iter += 1
        self.set_SLA()
        self.LAI = SLA_to_LAI(self.SLA, c_tau, self.leaf_to_shoot_ratio, self.X_s, self.X_ns)
        self.CAC = LAI_to_CAC(self.LAI)
    def print_attributes(self, *args):
        if args:  # If specific attribute names are provided
            for attr_name in args:
                if hasattr(self, attr_name):  # Check if the attribute exists
                    print(f"{attr_name}: {getattr(self, attr_name)}")
                else:
                    print(f"Attribute '{attr_name}' not found.")
        else:  # If no specific attribute names are provided, print all attributes
            for attr, value in self.__dict__.items():
                print(f"{attr}: {value}")
    def biomass_ode(self, X_ns, X_s,T_air, CO2_air, PAR_flux, PPFD, g_bnd, g_stm):
        CO2_ppm = CO2_air
        g_car = c_car_1*T_air**2 + c_car_2*T_air + c_car_3
        g_CO2 = 1/(1/g_bnd + 1/g_stm + 1/g_car)
        Gamma = c_Gamma*c_q10_Gamma**((T_air-20)/10)
        epsilon_biomass = c_epsilon_calibrated(PPFD, T_air) * (CO2_ppm-Gamma)/(CO2_ppm + 2*Gamma)
        f_phot_max = (epsilon_biomass*PAR_flux*g_CO2*c_w*(CO2_ppm-Gamma))/(epsilon_biomass*PAR_flux+g_CO2*c_w*(CO2_ppm-Gamma))
        f_phot = (1-np.exp(-c_K*c_lar*(1-c_tau)*X_s))*f_phot_max
        f_resp = (c_resp_sht*(1-c_tau)*X_s+c_resp_rt*c_tau*X_s)*c_q10_resp**((T_air-25)/10)

        r_gr = c_gr_max_calibrated(PPFD,T_air)*X_ns/(c_gamma*X_s+X_ns)*c_q10_gr**((T_air-20)/10)
        
        dX_ns = c_a*f_phot - r_gr*X_s - f_resp - (1-c_beta_calibrated(PPFD, T_air))/c_beta_calibrated(PPFD, T_air)*r_gr*X_s

        dX_s = r_gr*X_s

        self.f_phot_max = f_phot_max
        self.f_phot = f_phot
        self.f_resp = f_resp
        self.r_gr = r_gr
        return dX_ns, dX_s
    def transpiration_ode(self):
        return 0

    def crop_conditions(self, env_state, crop_state, climate, control_input):
        T_air, Chi_air, CO2_air = env_state
        X_ns, X_s = crop_state
        
        LAI = SLA_to_LAI(self.SLA, c_tau, self.leaf_to_shoot_ratio, X_s, X_ns)

        # Varying parameters
        T_out, Chi_out, DAT = climate
        self.DAT = DAT
        PAR_flux, PPFD ,wind_vel = control_input
        g_bnd = 1/aerodynamical_resistance_eq(wind_vel, LAI)
        g_stm = 1/stomatal_resistance_eq(PPFD)
        dNS_dt, dS_dt = self.biomass_ode(X_ns, X_s, T_air, CO2_air, PAR_flux, PPFD, g_bnd, g_stm)
        return np.array([dNS_dt, dS_dt])