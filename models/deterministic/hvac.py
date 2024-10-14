from typing import Dict, Any
from models.deterministic.utils import calculate_absolute_humidity
from CoolProp.HumidAirProp import HAPropsSI
from config.utils import get_attribute

class HVACModel:
    def __init__(self, config: Dict[str, Any]):
        # Constants
        self.p = get_attribute(config, 'p')
        self.c_air = get_attribute(config, 'c_air')
        self.rho_air = get_attribute(config, 'rho_air')
        self.Lambda = get_attribute(config, 'lambda')

        # Efficiency parameters
        self.eta_rot_T = get_attribute(config, 'eta_rot_T')
        self.eta_rot_Chi = get_attribute(config, 'eta_rot_Chi')

    def rotary_heat_exchanger(self, T_in, T_out, Chi_in, Chi_out, u_rot):
        T_rot = u_rot*self.eta_rot_T*(T_in - T_out) + T_out
        Chi_rot = u_rot*self.eta_rot_Chi*(Chi_in - Chi_out) + Chi_out

        return T_rot, Chi_rot
    
    def supply_fan(self, T_rot):
        T_fan = T_rot + 1

        return T_fan
    
    def cooling_and_dehumidification_coil(self, h_fan, W_fan, u_cool, u_sup):
        h_cool = h_fan - u_cool/(u_sup*self.rho_air)

        try: # Sensible cooling
            RH_out = HAPropsSI('R','H',h_cool/1000,'P',self.p,'W',W_fan/1000) # h_cool must be in kJ/kg for CoolProp
        except ValueError: # Dew point reached, sensible and latent cooling (condensation)
            RH_out = 1
        
        T_cool = HAPropsSI('T','H',h_cool/1000,'P',self.p,'R',RH_out)
        W_cool = HAPropsSI('W','H',h_cool/1000,'P',self.p,'R',RH_out)
        Chi_cool = (W_cool*1000)*self.rho_air

        return T_cool, Chi_cool
    
    def heating_coil(self, T_cool, u_heat, u_sup):
        T_heat = T_cool + u_heat/(u_sup*self.rho_air*self.c_air)

        return T_heat
    
    def humidifier(self, Chi_cool, u_humid, u_sup):
        Chi_humid = Chi_cool + u_humid/u_sup

        return Chi_humid
    
    def calculate_supply_air(self, state, control_inputs, data):    
        T_in, Chi_in, CO2_in, T_env, T_sup, Chi_sup, X_ns, X_s = state
        u_rot, u_sup, u_cool, u_heat, u_humid, u_c_inj, PPFD = control_inputs

        T_out = float(data[0])
        RH_out = float(data[1])
        Chi_out = calculate_absolute_humidity(self.p, T_out, RH_out)

        if u_sup == 0:
            return T_out, Chi_out, Chi_out

        # Convert Celsius to Kelvin
        T_in += 273.15
        T_out += 273.15

        # Calculate temperature and absolute humidity sequentially through the HVAC system
        T_rot, Chi_rot = self.rotary_heat_exchanger(T_in, T_out, Chi_in, Chi_out, u_rot)

        T_fan = self.supply_fan(T_rot)

        W_fan = Chi_rot/self.rho_air # grams of water per kg of dry air
        try:
            RH_fan = HAPropsSI('R','T',T_fan,'P',self.p,'W',W_fan/1000) # W_fan must be in kg/kg for CoolProp
        except ValueError:
            RH_fan = 1
        h_fan = HAPropsSI('H','T',T_fan,'P',self.p,'R',RH_fan)

        # Convert kJ/kg to J/kg
        h_fan = h_fan*1000 

        T_cool, Chi_cool = self.cooling_and_dehumidification_coil(h_fan, W_fan, u_cool, u_sup)

        T_heat = self.heating_coil(T_cool, u_heat, u_sup)

        Chi_humid = self.humidifier(Chi_cool, u_humid, u_sup)

        return T_heat - 273.15, Chi_humid, Chi_out