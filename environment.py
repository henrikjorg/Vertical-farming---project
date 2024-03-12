import numpy as np
from typing import Dict, Any
from config import *
from utilities import *
from scipy.optimize import minimize

class EnvironmentModel:
    def __init__(self, config: Dict[str, Any]):
        self.surface_temperature = 20  # Assuming constant initial surface temperature
        self.transpiration = 0
        self.T_air = config['init_T_air']
        self.RH = config['init_RH']
        self.CO2 = config['init_CO2']
        self.update_derived_values()

    def update_derived_values(self):
        self.Chi_sat = estimate_Chi_sat(self.T_air)
        self.VCD = self.Chi_sat * (100 - self.RH)  # Vapor pressure deficit, adjust formula as needed

    def update_values(self, T_air: float, RH: float, CO2: float):
        self.T_air = T_air
        self.RH = RH
        self.CO2 = CO2
        self.update_derived_values()


    def print_attributes(self, *args):
        if args:
            for attr_name in args:
                print(f"{attr_name}: {getattr(self, attr_name, 'Attribute not found')}")
        else:
            for attr, value in vars(self).items():
                print(f"{attr}: {value}")

 
    def find_transpiration(self, T_air, Chi_air, LAI, CAC, PAR_flux, r_a, r_s, rho_r):
        def energy_balance(surface_temperature, T_air, Chi_air, LAI, r_s, r_a, net_radiation):
            sensible_heat_flux = LAI * rho_a * c_specific_heat * (surface_temperature - T_air) / r_a
            Chi_surface = estimate_Chi_surface(T_air, surface_temperature)
            latent_heat_flux = LAI * lambda_water * (Chi_surface - Chi_air) / (r_s + r_a)
            return abs(net_radiation - sensible_heat_flux - latent_heat_flux)

        initial_guess = self.surface_temperature
        net_radiation = net_radiation_equation(PAR_flux, CAC, rho_r)
        args = (T_air, Chi_air, LAI, r_s, r_a, net_radiation)
        result = minimize(energy_balance, initial_guess, args=args, method='Nelder-Mead', options={'fatol': 1e-2})

        if not result.success:
            raise ValueError("Optimization failed to find a solution for surface temperature.")

        surface_temperature = result.x[0]
        self.surface_temperature = surface_temperature

        E = LAI * (estimate_Chi_surface(T_air, surface_temperature) - Chi_air) / (r_s + r_a)
        latent_heat_flux = E * lambda_water
        self.transpiration = E
        return surface_temperature, E, latent_heat_flux

    def temperature_ode(self, T_air, latent_heat_flux, T_out, PAR_flux, CAC):
        Q_cov = alpha_cov * (T_air - T_out)
        Q_trans = latent_heat_flux
        return 0  # Simplified for brevity

    def humidity_ode(self, Chi_air, Chi_out, E):
        return 0  # Simplified for brevity

    def co2_ode(self):
        return 0  # Simplified for brevity

    def env_conditions(self, env_state, crop_state, climate, control_input, crop_model):
        T_air, RH, CO2_air = env_state
        X_ns, X_s = crop_state
        self.update_values(T_air=T_air, RH=RH, CO2=CO2_air)
        Chi_air_sat = estimate_Chi_sat(T_air)
        Chi_air = Chi_air_sat * RH / 100
        LAI = crop_model.LAI
        CAC = crop_model.CAC
        T_out, Chi_out, DAT = climate
        PAR_flux, PPFD, wind_vel = control_input

        r_s = stomatal_resistance_eq(PPFD=PPFD)
        r_a = aerodynamical_resistance_eq(uninh_air_vel=wind_vel, LAI=LAI, leaf_diameter=crop_model.leaf_diameter)
        T_surface, E, latent_heat_flux = self.find_transpiration(T_air=T_air, Chi_air=Chi_air, LAI=LAI, CAC=CAC, PAR_flux=PAR_flux, r_a=r_a, r_s=r_s, rho_r=crop_model.rho_r)

        dT_air_dt = self.temperature_ode(T_air, latent_heat_flux, T_out, PAR_flux, CAC)
        dChi_air_dt = self.humidity_ode(Chi_air, Chi_out, E)
        dCO2_air_dt = self.co2_ode()

        return np.array([dT_air_dt, dChi_air_dt, dCO2_air_dt])
