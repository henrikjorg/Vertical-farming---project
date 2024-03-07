import numpy as np
from scipy.integrate import solve_ivp
from environment import EnvironmentModel
from crop import CropModel
from config import *


class Model:
    def __init__(self):
        self.env_model = EnvironmentModel()
        self.crop_model = CropModel()

    def model(self, t, y, climate, control_input):
        env_state = y[:len(env_states_info)]
        crop_state = y[len(env_states_info):]

        # Update models
        env_derivatives = self.env_model.env_conditions(env_state, crop_state, climate, control_input, crop_model = self.crop_model)
        plant_derivatives = self.crop_model.crop_conditions(env_state, crop_state,climate, control_input, crop_model = self.crop_model )

        # Combine derivatives
        return np.concatenate((env_derivatives, plant_derivatives), axis=None)
