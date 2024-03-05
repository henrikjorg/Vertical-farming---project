import numpy as np
from config import *
from scipy.interpolate import griddata

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

# Arrays of your experimental data
PPFD_values = np.array([200, 200, 200, 400, 400, 400, 750, 750, 750])
temperature_values = np.array([20, 24, 28, 20, 24, 28, 20, 24, 28])
optimal_c_epsilon_parameters = np.array([10.5,11,10.75,10.75,11.75,9.75,8,8.1,7]) * 10**(-6) 
optimal_c_gr_max_parameters = np.array([0.8,0.7,0.2,1,0.9,0.5,1.2,1,0.5]) * 10**(-6)
optimal_c_beta_parameters = np.array([0.4,0.4,0.42,0.4,0.4,0.4,0.4,0.4,0.4])

def c_epsilon_calibrated(PPFD, temperature):
    # Ensure PPFD and temperature are within the experimental range
    PPFD = max(200, min(PPFD, 750))  # Adjust PPFD to be within [200, 750]
    temperature = max(20, min(temperature, 28))  # Adjust temperature to be within [20, 28]
    
    # Points where you have data
    points = np.array([PPFD_values, temperature_values]).T
    # Interpolate (or effectively look up) for the adjusted PPFD and temperature
    c_epsilon = griddata(points, optimal_c_epsilon_parameters, (PPFD, temperature), method='linear')
    
    return c_epsilon
def c_gr_max_calibrated(PPFD, temperature):
    # Ensure PPFD and temperature are within the experimental range
    PPFD = max(200, min(PPFD, 750))  # Adjust PPFD to be within [200, 750]
    temperature = max(20, min(temperature, 28))  # Adjust temperature to be within [20, 28]
    
    # Points where you have data
    points = np.array([PPFD_values, temperature_values]).T
    # Interpolate (or effectively look up) for the adjusted PPFD and temperature
    c_gr_max_calculated = griddata(points, optimal_c_gr_max_parameters, (PPFD, temperature), method='linear')
    
    return c_gr_max_calculated

def c_beta_calibrated(PPFD, temperature):
    # Ensure PPFD and temperature are within the experimental range
    PPFD = max(200, min(PPFD, 750))  # Adjust PPFD to be within [200, 750]
    temperature = max(20, min(temperature, 28))  # Adjust temperature to be within [20, 28]
    
    # Points where you have data
    points = np.array([PPFD_values, temperature_values]).T
    # Interpolate (or effectively look up) for the adjusted PPFD and temperature
    c_beta_calculated = griddata(points, optimal_c_beta_parameters, (PPFD, temperature), method='linear')
    
    return c_beta_calculated

def DW_leaf(FW_shoot, DW, DW_content):
    leaves_to_shoot_ratio = 0.92 # As reported by Talbot
    return leaves_to_shoot_ratio * FW_shoot*DW*DW_content
def SLA_to_LAI(SLA, k, leaf_to_shoot_ratio,X_s,X_ns):
     return SLA*(1-k)*leaf_to_shoot_ratio*(X_s+X_ns)