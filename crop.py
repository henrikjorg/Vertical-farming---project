import numpy as np

# Modeling crop growth and plant transpiration.

class CropModel:
    def transpiration_ode(self):
        return 0
    
    def photosynthesis_ode(self):
        return 0
    
    def crop_conditions(self, env_state, crop_state):
        T_air, Chi_air, CO2_air = env_state
        Transp_crop, Photo_crop = crop_state

        dTransp_dt = self.transpiration_ode()
        dPhoto_dt = self.photosynthesis_ode()
        return np.array([dTransp_dt, dPhoto_dt])
    

        
