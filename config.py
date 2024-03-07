env_states_info = {
    'temperature': {
        'title': 'Temperature', 
        'unit': '°C', 
        'color': 'b',
        'init_value': 20.0},
    'humidity': {
        'title': 'Relative Humidity', 
        'unit': '%', 
        'color': 'g',
        'init_value': 75.0},
    'co2': {
        'title': 'CO2', 
        'unit': 'ppm', 
        'color': 'r',
        'init_value': 1200.0}, 
}

crop_states_info = {
    'X_ns': {
        'title': 'Non-structural dry weight', 
        'unit': 'g', 
        'color': 'g',
        'init_value': None}, # Weight is given for FW per plant in the initialization of the crop class 
    'X_s': {
    'title': 'Structural dry weight', 
    'unit': 'g', 
    'color': 'y',
    'init_value': 5},   # Weight is given for FW per plant in the initialization of the crop class 
}

# (TODO): Find correct parameters
# constant parameters
c_cap = 1000
alpha_cov = 5
g_e = 1
L = 1
h = 1
g_V = 1

# Crop tranpiration coefficients
#rho_r = 0.06 # reflection coefficient of the crop
lambda_water = 2.450 # Latent heat of vaporization of water J/kg
#l_d = 0.05 # Mean leaf diameter [m]
rho_a = 1.204 # Air density [kg/m^3]
c_specific_heat = 1005 # specific heat of air [J/g°C]
epsilon = 1 # Not correct

# Crop biomass coefficients
#c_a = 0.68 # Ratio of CH2O and CO2
#c_beta = 0.72 # for lettuce. Taken from Kacira. Van Henten used 0.8
#c_gr_max = 5*10**(-6) # [1/s]
#c_q10_gr = 1.6 # q10 growth factor
#c_gamma = 1
#c_resp_sht = 3.47*10**(-7) # [1/s] maintenance respiration coefficient for shoot at 25 deg
#c_resp_rt = 1.16*10**(-7) # [1/s] maintenance respiration coef for root
#c_q10_resp = 2
#c_tau = 0.15 # for lettuce grown in soil
#c_K = 0.9#0.66 As reported by Talbot (0.9 according to van Henten for planophile plants like lettuce)
#c_lar = 75*10**(-3) # [m^2/g] structural leaf area ratio (assumed constant)
#c_w = 1.893*10**(-3) # [g/m^3] density of CO2. Taken from Kacira. Van Henten had 1.83 (i think)
#c_Gamma = 71.5 # [ppm] CO2 compensation point at 20 deg. Kacira uses 71.5, which he obtains from tuning. Van Henten uses 40
#c_q10_Gamma = 2
#c_epsilon = 17*10**(-6) # [g/J] light use efficiency at very high CO2 concentrations
#c_car_1 = -1.32*10**(-5)
#c_car_2 = 5.94*10**(-4)
#c_car_3 = -2.64*10**(-3)


