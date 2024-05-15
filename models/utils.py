import numpy as np
from config import *
from scipy.interpolate import griddata
import json

def net_radiation_equation(PAR_flux, CAC, rho_r):
    """
    Calculates the net radiation absorbed by a cultivation area.

    Parameters:
    - PAR_flux (W/m²): Photosynthetically Active Radiation flux, the amount of usable light energy received per unit area.
    - CAC (unitless): Cultivation Area Coefficient, the ratio of projected leaf area to cultivation area, indicating the proportion of the cultivation area covered by leaves.
    - rho_r (unitless): Reflectivity of the canopy, the proportion of PAR reflected by the canopy.

    Returns:
    - Net radiation absorbed by the cultivation area (W/m²): The amount of PAR flux absorbed by the leaf area after accounting for reflectivity, considering the entire cultivation area.
    """
    return (1 - rho_r) * PAR_flux * CAC


#def biomass_to_LAI(X_s):
     # This is the LAI estimated in Van Henten, but it is replaced by Talbot's SLA_to_LAI function
#     return (1-c_tau)*c_lar*X_s
def LAI_to_CAC(LAI, k=0.5):
    """
    Converts Leaf Area Index (LAI) to Canopy Absorption Coefficient (CAC).

    Parameters:
    - LAI (m²/m²): Leaf Area Index, the area of leaves per unit ground area.
    - k (unitless): Extinction coefficient for the canopy (default is 0.5).

    Returns:
    - CAC (unitless): Cultivation Area Coefficient, the ratio of projected leaf area to cultivation area, indicating the proportion of the cultivation area covered by leaves.
    """
    return 1 - np.exp(-k * LAI)

def stomatal_resistance_eq(PPFD):
    """
    Calculates stomatal resistance based on Photosynthetic Photon Flux Density (PPFD).

    Parameters:
    - PPFD (μmol/m²/s): Photosynthetic Photon Flux Density, the amount of photosynthetically active photons.

    Returns:
    - Stomatal resistance (s/m): The resistance to CO2 and water vapor flux through the stomata.
    """
    return 60 * (1500 + PPFD) / (200 + PPFD)

def aerodynamical_resistance_eq(uninh_air_vel, LAI, leaf_diameter):
    """
    Calculates aerodynamic resistance based on air velocity, mean leaf diameter and Leaf Area Index (LAI).

    Parameters:
    - uninh_air_vel (m/s): Uninhibited air velocity.
    - LAI (m²/m²): Leaf Area Index, the area of leaves per unit ground area.
    -leaf_diameter (m): Mean leaf diameter.
    Returns:
    - Aerodynamic resistance (s/m): The resistance to heat and vapor flux from the canopy to the atmosphere.
    """
    return 350 * np.sqrt((leaf_diameter / uninh_air_vel)) * 1 / (LAI + 0.001)

# Environmental values from Carotti, with optimal Values written off the plots of Talbot
PPFD_values = np.array([200, 200, 200, 400, 400, 400, 750, 750, 750])
temperature_values = np.array([20, 24, 28, 20, 24, 28, 20, 24, 28])
optimal_c_epsilon_parameters = np.array([10.5,11,10.75,10.75,11.75,9.75,8,8.1,7]) * 10**(-6) 
optimal_c_gr_max_parameters = np.array([0.8,0.7,0.2,1,0.9,0.5,1.2,1,0.5]) * 10**(-6)
optimal_c_beta_parameters = np.array([0.4,0.4,0.42,0.4,0.4,0.4,0.4,0.4,0.4])

def c_epsilon_calibrated(PPFD, temperature, use_calibrated = False):
    if not use_calibrated:
        return 17e-6
    # Ensure PPFD and temperature are within the experimental range
    PPFD_temp = max(200, min(PPFD, 750))  # Adjust PPFD to be within [200, 750]
    temperature_temp = max(20, min(temperature, 28))  # Adjust temperature to be within [20, 28]
    
    # Points where you have data
    points = np.array([PPFD_values, temperature_values]).T
    # Interpolate (or effectively look up) for the adjusted PPFD and temperature
    c_epsilon = griddata(points, optimal_c_epsilon_parameters, (PPFD_temp, temperature_temp), method='linear')
    return c_epsilon
def c_gr_max_calibrated(PPFD, temperature, use_calibrated = False):
    if not use_calibrated:
        return 5*10**(-6)
    # Ensure PPFD and temperature are within the experimental range
    PPFD_temp = max(200, min(PPFD, 750))  # Adjust PPFD to be within [200, 750]
    temperature_temp = max(20, min(temperature, 28))  # Adjust temperature to be within [20, 28]
    
    # Points where you have data
    points = np.array([PPFD_values, temperature_values]).T
    # Interpolate (or effectively look up) for the adjusted PPFD and temperature
    c_gr_max_calculated = griddata(points, optimal_c_gr_max_parameters, (PPFD_temp, temperature_temp), method='linear')
   
    return c_gr_max_calculated
    
        
def c_beta_calibrated(PPFD, temperature, use_calibrated = False):
    if not use_calibrated:
        return 0.72
    # Ensure PPFD and temperature are within the experimental range
    PPFD_temp = max(200, min(PPFD, 750))  # Adjust PPFD to be within [200, 750]
    temperature_temp = max(20, min(temperature, 28))  # Adjust temperature to be within [20, 28]
    
    # Points where you have data
    points = np.array([PPFD_values, temperature_values]).T
    # Interpolate (or effectively look up) for the adjusted PPFD and temperature
    c_beta_calculated = griddata(points, optimal_c_beta_parameters, (PPFD_temp, temperature_temp), method='linear')
    return c_beta_calculated

def SLA_to_LAI(SLA, c_tau, leaf_to_shoot_ratio, X_s, X_ns):
    """
    Calculates the Leaf Area Index (LAI) from specific leaf area, biomass allocation parameters, and total biomass.

    Parameters:
    - SLA (m²/kg): Specific Leaf Area, the area of leaves per unit of dry leaf mass.
    - c_tau (unitless): Ratio of the root dry weight to the total dry weight (0 <= k <= 1).
    - leaf_to_shoot_ratio (unitless): Ratio of leaf biomass to total shoot biomass.
    - X_s (kg): Dry mass of structural (non-leafy) parts of the plant.
    - X_ns (kg): Dry mass of non-structural (leafy) parts of the plant.

    Returns:
    - LAI (m²/m²): Leaf Area Index, the total one-sided area of leaf tissue per unit ground surface area.
    """
    return SLA * (1 - c_tau) * leaf_to_shoot_ratio * (X_s + X_ns)


###########

def estimate_Chi_sat_in_hPa(T):
    """
    Calculate the saturation vapor pressure (e_s) over a flat surface of water
    at temperature T using the Tetens formula.
    
    Parameters:
    - T: Temperature in degrees Celsius
    
    Returns:
    - e_s: Saturation vapor pressure in hPa (same as millibars)
    """
    e_s = 6.112 * np.exp((17.67 * T) / (T + 243.5))
    return e_s
def clausius_clapeyron_equation(T, e_s):
    """
    Calculate the rate of change of saturation vapor pressure with temperature
    using the Clausius-Clapeyron equation.
    
    Parameters:
    - T: Temperature in degrees Celsius
    - e_s: Saturation vapor pressure at temperature T in hPa
    
    Returns:
    - d_es_dT: The rate of change of saturation vapor pressure with temperature in hPa/K
    """
    L = 2.501e6  # Latent heat of vaporization (J/kg)
    R_v = 461.5  # Specific gas constant for water vapor (J/(kg·K))
    d_es_dT = (L * e_s) / (R_v * (T + 273.15)**2)
    return d_es_dT
def estimate_Chi_sat(T):
    """
    Calculate the density of water vapor (rho) in g/m^3 at saturation (100% relative humidity)
    at temperature T using the Ideal Gas Law and the Tetens formula for vapor pressure.
    
    Parameters:
    - T: Temperature in degrees Celsius
    
    Returns:
    - rho: Density of water vapor at saturation in g/m^3
    """
    # Convert e_s from hPa to Pa for the calculation
    e_s_pa = estimate_Chi_sat_in_hPa(T) * 100
    M_w = 18.01528  # Molar mass of water (g/mol)
    R = 8.31447  # Universal gas constant (J/(mol·K))
    rho = (e_s_pa * M_w) / (R * (T + 273.15))
    return rho

def estimate_dChi_dT(T):
    """
    Calculate the rate of change of water vapor density with temperature in g/m^3 per degree Celsius
    using the Clausius-Clapeyron equation and the differentiation of the Ideal Gas Law as it applies
    to saturation vapor pressure and vapor density.
    
    Parameters:
    - T: Temperature in degrees Celsius
    
    Returns:
    - d_rho_dT: Rate of change of water vapor density with temperature in g/m^3 per degree Celsius
    """
    e_s_pa = estimate_Chi_sat_in_hPa(T) * 100  # Saturation vapor pressure in Pa
    d_es_dT_pa = clausius_clapeyron_equation(T, e_s_pa / 100) * 100  # Rate of change of e_s in Pa/K
    M_w = 18.01528  # Molar mass of water (g/mol)
    R = 8.31447  # Universal gas constant (J/(mol·K))
    
    # Differentiate rho with respect to T
    d_rho_dT = (d_es_dT_pa * M_w) / (R * (T + 273.15)) - (e_s_pa * M_w) / (R * (T + 273.15)**2)
    
    return d_rho_dT


def estimate_Chi_surface(air_temperature, surface_temperature):
    """
    Estimates the vapor concentration of a surface based on the hypothesis that
    a transpiring surface is saturated at its temperature, applying only the first
    (linear) term of Taylor's expansion of the saturation hypothesis.
    
    Parameters:
    - air_temperature: Air temperature in degrees Celsius
    - surface_temperature: Surface temperature in degrees Celsius
    
    Returns:
    - vapor_concentration_surface: Estimated vapor concentration at the surface in g/m^3
    """

    # Calculate the saturation vapor density at the air temperature
    rho_sat_air = estimate_Chi_sat(air_temperature)
    
    # Calculate the slope of the saturation curve at the air temperature
    d_rho_dT_air = estimate_dChi_dT(air_temperature)
    
    # Calculate the temperature difference between the surface and the air
    delta_T = surface_temperature - air_temperature
    
    # Apply the first term of Taylor's expansion to estimate the change in vapor concentration
    delta_rho = d_rho_dT_air * delta_T
    
    # Estimate the vapor concentration based on air temperature
    vapor_concentration_air_based = rho_sat_air + delta_rho
    
    return vapor_concentration_air_based

def load_config(file_path: str) -> dict:
    """Load configuration from a JSON file."""
    if file_path == 'opt_config.json' or file_path == 'opt_config_casadi.json':
        with open('mpc_optimization/' + file_path, 'r') as file:
            return json.load(file)
    else:
        with open(file_path, 'r') as file:
            return json.load(file)