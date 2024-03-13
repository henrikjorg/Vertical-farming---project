import numpy as np
from config import *
from utilities import *
from typing import Dict, Any

class CropModel:
    def __init__(self, config: Dict[str, Any]):
        # Basic configuration parameters
        self.ideal_temp: float = config.get('ideal_temp', 24)
        self.ideal_PPFD: float = config.get('ideal_PPFD', 400)
        self.init_FW_per_plant: float = config.get('init_FW_per_plant', 1)
        self.SLA: float = config.get('SLA', 0.03)
        self.plant_density: float = config.get('plant_density', 25)

        # Physiological parameters
        self.structural_to_nonstructural: float = config.get('structural_to_nonstructural', 0.75)
        self.dry_weight_fraction: float = config.get('dry_weight_fraction', 0.05)
        self.leaf_to_shoot_ratio: float = config.get('leaf_to_shoot_ratio', 0.92)
        self.rho_r: float = config.get('rho_r', 0.06)
        self.leaf_diameter: float = config.get('leaf_diameter', 0.11)

        # Growth coefficients and parameters
        self.c_tau: float = config.get('c_tau', 0.15)
        self.c_q10_gr: float = config.get('c_q10_gr', 1.6)
        self.c_gamma: float = config.get('c_gamma', 1)
        self.c_gr_max: float = config.get('c_gr_max', 5e-6)
        self.c_beta: float = config.get('c_beta', 0.72)
        self.c_a: float = config.get('c_a', 0.68)
        self.c_resp_sht: float = config.get('c_resp_sht', 3.47e-7)
        self.c_resp_rt: float = config.get('c_resp_rt', 1.16e-7)
        self.c_q10_resp: float = config.get('c_q10_resp', 2)
        self.c_K: float = config.get('c_K', 0.9)
        self.c_lar: float = config.get('c_lar', 75e-3)
        self.c_w: float = config.get('c_w', 1.893e-3)
        self.c_Gamma: float = config.get('c_Gamma', 71.5)
        self.c_q10_Gamma: float = config.get('c_q10_Gamma', 2)
        self.c_car_1: float = config.get('c_car_1', -1.32e-5)
        self.c_car_2: float = config.get('c_car_2', 5.94e-4)
        self.c_car_3: float = config.get('c_car_3', -2.64e-3)

        # Initialize dynamic attributes
        self.DAT: int = 0  # Days After Transplanting
        self.update_dynamic_attributes()

    def update_dynamic_attributes(self):
        self.fresh_weight_shoot_per_plant: float = self.init_FW_per_plant
        self.fresh_weight_shoot: float = self.fresh_weight_shoot_per_plant * self.plant_density
        self.dry_weight: float = self.fresh_weight_shoot * self.dry_weight_fraction / (1 - self.c_tau)
        self.X_ns: float = self.dry_weight * (1 - self.structural_to_nonstructural)
        self.X_s: float = self.dry_weight * self.structural_to_nonstructural
        self.LAI: float = SLA_to_LAI(SLA=self.SLA, c_tau=self.c_tau, leaf_to_shoot_ratio=self.leaf_to_shoot_ratio, X_s=self.X_s, X_ns=self.X_ns)
        self.CAC: float = LAI_to_CAC(self.LAI)

    def set_SLA(self, temp: float = 24, PPFD: float = 400):
        # Adjust SLA based on temperature and PPFD, implement as needed
        pass

    def set_fresh_weight_shoot(self):
        self.fresh_weight_shoot = self.dry_weight * (1 - self.c_tau) / self.dry_weight_fraction
        self.fresh_weight_shoot_per_plant = self.fresh_weight_shoot / self.plant_density

    def update_values(self, X_ns: float, X_s: float):
        self.X_ns = X_ns
        self.X_s = X_s
        self.dry_weight = X_ns + X_s
        self.set_fresh_weight_shoot()
        self.set_SLA()
        self.LAI = SLA_to_LAI(SLA=self.SLA, c_tau=self.c_tau, leaf_to_shoot_ratio=self.leaf_to_shoot_ratio, X_s=self.X_s, X_ns=self.X_ns)
        self.CAC = LAI_to_CAC(self.LAI)

    def print_attributes(self, *args):
        if args:
            for attr_name in args:
                print(f"{attr_name}: {getattr(self, attr_name, 'Attribute not found')}")
        else:
            for attr, value in vars(self).items():
                print(f"{attr}: {value}")
    def biomass_ode(self, X_ns: float, X_s: float, T_air: float, CO2_air: float, PAR_flux: float, PPFD: float, g_bnd: float, g_stm: float):
        CO2_ppm = CO2_air
        g_car = self.c_car_1 * T_air**2 + self.c_car_2 * T_air + self.c_car_3
        g_CO2 = 1 / (1 / g_bnd + 1 / g_stm + 1 / g_car)
        Gamma = self.c_Gamma * self.c_q10_Gamma ** ((T_air - 20) / 10)
        epsilon_biomass = c_epsilon_calibrated(PPFD, T_air) * (CO2_ppm - Gamma) / (CO2_ppm + 2 * Gamma)
        f_phot_max = (epsilon_biomass * PAR_flux * g_CO2 * self.c_w * (CO2_ppm - Gamma)) / (epsilon_biomass * PAR_flux + g_CO2 * self.c_w * (CO2_ppm - Gamma))
        f_phot = (1 - np.exp(-self.c_K * self.LAI)) * f_phot_max
        self.f_phot = f_phot
        self.p_phot_max = f_phot_max
        self.f_phot_converted = f_phot / 1.8015e-5
        f_resp = (self.c_resp_sht * (1 - self.c_tau) * X_s + self.c_resp_rt * self.c_tau * X_s) * self.c_q10_resp ** ((T_air - 25) / 10)
        self.f_resp = f_resp
        r_gr = c_gr_max_calibrated(PPFD, T_air) * X_ns / (self.c_gamma * X_s + X_ns) * self.c_q10_gr ** ((T_air - 20) / 10)
        dX_ns = self.c_a * f_phot - r_gr * X_s - f_resp - (1 - c_beta_calibrated(PPFD, T_air)) / c_beta_calibrated(PPFD, T_air) * r_gr * X_s
        dX_s = r_gr * X_s

        return dX_ns, dX_s

    def crop_conditions(self, env_state: tuple, crop_state: tuple, climate: tuple, control_input: tuple, crop_model):
        T_air, RH, CO2_air = env_state
    
        X_ns, X_s = crop_state
        self.update_values(X_ns, X_s)
        # Assume additional necessary logic and calculations are implemented here

        PAR_flux, PPFD, wind_vel = control_input
        
        LAI = SLA_to_LAI(SLA=self.SLA, c_tau=self.c_tau, leaf_to_shoot_ratio=self.leaf_to_shoot_ratio, X_s=X_s, X_ns=X_ns)
        g_bnd = 1 / aerodynamical_resistance_eq(uninh_air_vel=wind_vel, LAI=LAI, leaf_diameter=self.leaf_diameter)
        g_stm = 1 / stomatal_resistance_eq(PPFD=PPFD)
        
        dNS_dt, dS_dt = self.biomass_ode(X_ns, X_s, T_air, CO2_air, PAR_flux, PPFD, g_bnd, g_stm)
        return np.array([dNS_dt, dS_dt])
