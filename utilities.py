import numpy as np
from config import *
def saturated_vapor_concentration(T):
        """
        Calculate the saturated vapor concentration of air (Chi*) at a given temperature.
        
        Parameters:
        T (float): Temperature in degrees Celsius.

        Returns:
        float: Saturated vapor concentration in g/m^3.
        """
        # Constants
        R = 287.058  # Specific gas constant for dry air, J/(kg·K)
        # Calculate saturated vapor pressure in kPa using the simplified formula
        es = 0.6108 * np.exp((17.27 * T) / (T + 237.3))
        
        # Convert vapor pressure to concentration using the ideal gas law
        # Note: es needs to be converted from kPa to Pa by multiplying by 1000
        chi_star = (es * 1000) / (R * (T + 273.15))
        
        # Convert kg/m^3 to g/m^3 by multiplying by 1000
        chi_star_g_m3 = chi_star * 1000
        
        return chi_star_g_m3
def slope_of_saturation_vapor_pressure_curve(T):
    """
    Calculate the slope of the saturation vapor pressure curve (epsilon) at a given temperature.
    
    Parameters:
    T (float): Temperature in degrees Celsius.
    
    Returns:
    float: Slope of the saturation vapor pressure curve at temperature T, in kPa/°C.
    """
    # Calculate the slope of the saturation vapor pressure curve using the empirical formula
    epsilon = (4098 * (0.6108 * np.exp((17.27 * T) / (T + 237.3)))) / ((T + 237.3) ** 2)
    
    return epsilon
def net_radiation_equation(PAR_flux, CAC):
    return  (1-rho_r)*PAR_flux*CAC
def biomass_to_LAI(X_s):
     return (1-c_tau)*c_lar*X_s
def LAI_to_CAC(LAI, k=0.5):
     return 1 - np.exp(-k * LAI)
def stomatal_resistance_eq(PPFD):
     return 60*(1500+PPFD)/(200+PPFD) # stomatal resistance
        
def aerodynamical_resistance_eq(wind_vel, LAI):
    return 350 * np.sqrt((l_d/wind_vel))*1/LAI # aerodynamical resistance