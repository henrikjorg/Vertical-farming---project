import numpy as np
from models.climate import ClimateModel
from models.crop import CropModel
from config.utils import get_attribute

class Model:
    def __init__(self, config):
        self.climate_model = ClimateModel(config)
        self.crop_model = CropModel(config)

        # Set references between models
        self.climate_model.set_crop_model(self.crop_model)
        self.crop_model.set_climate_model(self.climate_model)

        # Set configuration attributes
        self.heating_capacity = get_attribute(config, 'heating_capacity')
        self.cooling_capacity = get_attribute(config, 'cooling_capacity')
        self.u_fan_max = get_attribute(config, 'u_fan_max')
        self.u_humid_max = get_attribute(config, 'u_humid_max')
        self.u_c_inj_max = get_attribute(config, 'u_c_inj_max')

    def denormalize_action(self, action):
        u_rot = action[0]
        u_fan = action[1]*self.u_fan_max
        u_cool = action[2]*self.cooling_capacity
        u_heat = action[3]*self.heating_capacity
        u_humid = action[4]*self.u_humid_max
        u_c_inj = action[5]*self.u_c_inj_max

        return np.array([u_rot, u_fan, u_cool, u_heat, u_humid, u_c_inj])

    def print_attributes(self):
        self.climate_model.print_attributes("T_in", "Chi_in", "CO2_in", "T_env", "T_sup", "Chi_sup")
        self.crop_model.print_attributes("X_ns", "X_s")

    def ODEs(self, t, state, control_input, external_input, hvac_input, ignore_environment = False):
        # Update models
        if ignore_environment:
            climate_derivatives = np.zeros(6)
        else:
            climate_derivatives = self.climate_model.combined_ODE(state, control_input, external_input, hvac_input)
        
        crop_derivatives = self.crop_model.combined_ODE(state, control_input, external_input)

        # Combine derivatives
        return np.concatenate((climate_derivatives, crop_derivatives), axis=None)
