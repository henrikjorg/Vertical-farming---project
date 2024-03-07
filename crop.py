import numpy as np
from config import *
from utilities import *

# Modeling crop growth and plant transpiration.

class CropModel:
    """
    A model to simulate crop growth and plant transpiration incorporating environmental,
    physiological, and management variables to predict plant development and water usage.
    
    Parameters:
        ideal_temp (float): Optimal temperature for crop growth, in degrees Celsius.
        ideal_PPFD (float): Optimal Photosynthetic Photon Flux Density for crop growth, in μmol m⁻² s⁻¹.
        init_FW_per_plant (float): Initial fresh weight per plant, in grams.
        structural_to_nonstructural (float): Ratio of structural biomass to non-structural biomass.
        SLA (float): Specific Leaf Area, leaf area per unit dry weight, in m²/kg.
        dry_weight_fraction (float): Fraction of the shoot's dry weight relative to its fresh weight.
        leaf_to_shoot_ratio (float): Ratio of leaf biomass to total shoot biomass.
        plant_density (float): Number of plants per square meter.
        rho_r (float): Reflectivity of the canopy, unitless.
        leaf_diameter (float): Average diameter of leaves, in meters.
        c_tau (float): Fraction of total biomass allocated to non-leafy parts.
        c_q10_gr (float): Q10 temperature coefficient for growth rate.
        c_gamma (float), c_gr_max (float), c_beta (float), c_a (float), c_resp_sht (float),
        c_resp_rt (float), c_q10_resp (float), c_K (float), c_lar (float), c_w (float),
        c_Gamma (float), c_q10_Gamma (float), c_car_1 (float), c_car_2 (float), c_car_3 (float):
            Coefficients and parameters used in the biomass ode and crop conditions functions.
    
    Attributes:
        DAT (int): Days After Transplanting, used to track the age of the crop.
        fresh_weight_shoot_per_plant (float): Fresh weight of the shoot per plant, updated during simulation.
        fresh_weight_shoot (float): Total fresh weight of the shoot for all plants, in grams.
        dry_weight (float): Dry weight of the shoot, calculated from fresh weight, in grams.
        X_ns (float): Non-structural biomass, in grams.
        X_s (float): Structural biomass, in grams.
        SLA (float): Specific Leaf Area, updated during simulation if necessary.
        LAI (float): Leaf Area Index, calculated from SLA and biomass data.
        CAC (float): Cultivation Area Coefficient, derived from LAI.
        coeffs (dict): A dictionary storing all model coefficients and parameters for easy access and modification.
    
    Methods:
        set_SLA(temp, PPFD): Adjusts the Specific Leaf Area based on temperature and PPFD, to be implemented.
        set_fresh_weight_shoot(): Updates the fresh weight of the shoot based on dry weight and model coefficients.
        update_values(X_ns, X_s): Updates biomass values (non-structural and structural) and recalculates dependent model attributes.
        print_attributes(*args): Prints the specified model attributes. If no attributes are specified, prints all attributes.
        biomass_ode(X_ns, X_s, T_air, CO2_air, PAR_flux, PPFD, g_bnd, g_stm): Defines the differential equations for biomass growth based on environmental conditions and physiological parameters.
        crop_conditions(env_state, crop_state, climate, control_input, crop_model): Updates the model based on current environmental conditions, crop state, and external inputs, returning the rate of change of non-structural and structural biomass.
    """
    def __init__(self, ideal_temp=24, ideal_PPFD=400, init_FW_per_plant=1, structural_to_nonstructural = 0.75, SLA = 0.3, dry_weight_fraction = 0.05, leaf_to_shoot_ratio = 0.92
                 , plant_density=25, rho_r=0.06, leaf_diameter=0.11, c_tau = 0.15, c_q10_gr=1.6, c_gamma = 1, c_gr_max = 5*10**(-6), c_beta = 0.72, c_a = 0.68, c_resp_sht = 3.47*10**(-7),
                 c_resp_rt = 1.16*10**(-7), c_q10_resp = 2, c_K = 0.9, c_lar = 75*10**(-3), c_w = 1.893*10**(-3), c_Gamma = 71.5, c_q10_Gamma = 2,
                 c_car_1 = -1.32*10**(-5),c_car_2 = 5.94*10**(-4),c_car_3 = -2.64*10**(-3)):
        self.coeffs = {
              'rho_r': rho_r,
              'structural_to_nonstructural': structural_to_nonstructural,
              'dry_weight_fraction': dry_weight_fraction,
              'leaf_to_shoot_ratio': leaf_to_shoot_ratio,
              'plant_density': plant_density, # Carotti had a density of 25/m^2
              'ideal_temp': ideal_temp,
              'ideal_PPFD': ideal_PPFD,
              'leaf_diameter': leaf_diameter,
              'c_tau': c_tau,
              'c_q10_gr': c_q10_gr, # q10 growth factor
              'c_q10_resp' : c_q10_resp,
              'c_K' : c_K, #0.66 As reported by Talbot (0.9 according to van Henten for planophile plants like lettuce)
              'c_gamma' : c_gamma,
              'c_lar' : c_lar, # [m^2/g] structural leaf area ratio (assumed constant)
              'c_w' : c_w,   # [g/m^3] density of CO2. Taken from Kacira. Van Henten had 1.83 (i think)
              'c_Gamma' : c_Gamma, # [ppm] CO2 compensation point at 20 deg. Kacira uses 71.5, which he obtains from tuning. Van Henten uses 40
              'c_q10_Gamma' : c_q10_Gamma,
              'c_car_1' : -1.32*10**(-5),
              'c_car_2' : 5.94*10**(-4),
              'c_car_3' : -2.64*10**(-3),



              'c_gr_max' : c_gr_max, # [1/s]
              'c_beta' : c_beta, # 0.72 in Kacira, 0.8 in van Henten, approx 0.4 in Talbot
              'c_a' : c_a, # Ratio of CH2O and CO2
              'c_resp_sht' : c_resp_sht, # [1/s] maintenance respiration coefficient for shoot at 25 deg
              'c_resp_rt' :c_resp_rt # [1/s] maintenance respiration coef for root


        }
        self.DAT = 0
        self.fresh_weight_shoot_per_plant = init_FW_per_plant
        self.fresh_weight_shoot = self.fresh_weight_shoot_per_plant*plant_density
        self.dry_weight = self.fresh_weight_shoot*dry_weight_fraction/(1-c_tau) 
        self.X_ns = self.dry_weight*(1-structural_to_nonstructural)
        self.X_s = self.dry_weight*structural_to_nonstructural
        self.SLA = SLA
        self.LAI = SLA_to_LAI(SLA=self.SLA, c_tau=c_tau, leaf_to_shoot_ratio=leaf_to_shoot_ratio, X_s=self.X_s, X_ns=self.X_ns)
        self.CAC = LAI_to_CAC(self.LAI)
    def set_SLA(self, temp=24, PPFD=400):
        # TODO
        return
    def set_fresh_weight_shoot(self):
        # c_tau, or the ratio of root dry weight to the total dry weight for lettuce grown in soil
        self.fresh_weight_shoot = self.dry_weight*(1-self.coeffs['c_tau'])/self.coeffs['dry_weight_fraction']
        self.fresh_weight_shoot_per_plant = self.fresh_weight_shoot/self.coeffs['plant_density']
    def update_values(self, X_ns,X_s):
        #Updates the dynamic attributes
        self.X_ns = X_ns
        self.X_s = X_s
        self.dry_weight = X_ns+X_s
        self.set_fresh_weight_shoot()
        self.set_SLA()
        self.LAI = SLA_to_LAI(self.SLA, self.coeffs['c_tau'], self.coeffs['leaf_to_shoot_ratio'], self.X_s, self.X_ns)
        self.CAC = LAI_to_CAC(self.LAI)
    def print_attributes(self, *args):
        # Prints the specified attributes. Prints all attributes if none are selected
        if args:  # If specific attribute names are provided
            for attr_name in args:
                if hasattr(self, attr_name):  # Check if the attribute exists
                    print(f"{attr_name}: {getattr(self, attr_name)}")
                else:
                    print(f"Attribute '{attr_name}' not found.")
        else:  # If no specific attribute names are provided, print all attributes
            for attr, value in self.__dict__.items():
                print(f"{attr}: {value}")
    def biomass_ode(self, X_ns, X_s,T_air, CO2_air, PAR_flux, PPFD, g_bnd, g_stm):
        CO2_ppm = CO2_air
        g_car = self.coeffs['c_car_1']*T_air**2 + self.coeffs['c_car_2']*T_air + self.coeffs['c_car_3']
        g_CO2 = 1/(1/g_bnd + 1/g_stm + 1/g_car)
        Gamma = self.coeffs['c_Gamma']*self.coeffs['c_q10_Gamma']**((T_air-20)/10)
        epsilon_biomass = c_epsilon_calibrated(PPFD, T_air) * (CO2_ppm-Gamma)/(CO2_ppm + 2*Gamma)
        f_phot_max = (epsilon_biomass*PAR_flux*g_CO2*self.coeffs['c_w']*(CO2_ppm-Gamma))/(epsilon_biomass*PAR_flux+g_CO2*self.coeffs['c_w']*(CO2_ppm-Gamma))
        f_phot = (1-np.exp(-self.coeffs['c_K']*self.coeffs['c_lar']*(1-self.coeffs['c_tau'])*X_s))*f_phot_max
        f_resp = (self.coeffs['c_resp_sht']*(1-self.coeffs['c_tau'])*X_s+self.coeffs['c_resp_rt']*self.coeffs['c_tau']*X_s)*self.coeffs['c_q10_resp']**((T_air-25)/10)

        r_gr = c_gr_max_calibrated(PPFD,T_air)*X_ns/(self.coeffs['c_gamma']*X_s+X_ns)*self.coeffs['c_q10_gr']**((T_air-20)/10)     
        dX_ns = self.coeffs['c_a']*f_phot - r_gr*X_s - f_resp - (1-c_beta_calibrated(PPFD, T_air))/c_beta_calibrated(PPFD, T_air)*r_gr*X_s

        dX_s = r_gr*X_s

        self.f_phot_max = f_phot_max
        self.f_phot = f_phot
        self.f_resp = f_resp
        self.r_gr = r_gr
        return dX_ns, dX_s

    def crop_conditions(self, env_state, crop_state, climate, control_input, crop_model):
        T_air, Chi_air, CO2_air = env_state
        X_ns, X_s = crop_state
        
        LAI = SLA_to_LAI(self.SLA, self.coeffs['c_tau'], self.coeffs['leaf_to_shoot_ratio'], X_s, X_ns)

        # Varying parameters
        T_out, Chi_out, DAT = climate
        self.DAT = DAT
        PAR_flux, PPFD ,wind_vel = control_input
        g_bnd = 1/aerodynamical_resistance_eq(uninh_air_vel=wind_vel, LAI=LAI, leaf_diameter=crop_model.coeffs['leaf_diameter'])
        g_stm = 1/stomatal_resistance_eq(PPFD=PPFD)
        dNS_dt, dS_dt = self.biomass_ode(X_ns, X_s, T_air, CO2_air, PAR_flux, PPFD, g_bnd, g_stm)
        return np.array([dNS_dt, dS_dt])