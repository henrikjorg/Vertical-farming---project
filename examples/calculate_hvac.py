import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Set path for running local script
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Ignore tensorflow warning

from config.utils import load_config
from config.utils import get_attribute

from CoolProp.HumidAirProp import HAPropsSI
import pandas as pd
from tqdm import tqdm
import csv
import datetime
import os

# Import config parameters
config = load_config('../config/')

Lambda = get_attribute(config, 'lambda')
eta_rot_T = get_attribute(config, 'eta_rot_T')
eta_rot_Chi = get_attribute(config, 'eta_rot_Chi')
u_sup = get_attribute(config, 'u_sup_max')
rho_air = get_attribute(config, 'rho_air')
c_air = get_attribute(config, 'c_air')
Lambda = get_attribute(config, 'lambda')

u_ext = u_sup

# Import data from CSV file
df = pd.read_csv('../render/csv/winter_LED52_simulation.csv')
dates = df['Date'].to_numpy()
P_lights = df['P_light'].to_numpy()
T_ins = df['T_in'].to_numpy()
T_outs = df['T_out'].to_numpy()
Chi_ins = df['Chi_in'].to_numpy()
Chi_outs = df['Chi_out'].to_numpy()
desired_T_sups = df['T_sup'].to_numpy()
desired_Chi_sups = df['Chi_sup'].to_numpy()

# Initialize arrays to store results
new_dates = []
new_P_lights = []
u_rots = []
Q_cools = []
Q_heats = []
Q_humids = []

for i in tqdm(range(0, len(T_ins) + 1, 60*60)):
    T_in = T_ins[i]
    T_out = T_outs[i]
    Chi_in = Chi_ins[i]
    Chi_out = Chi_outs[i]
    desired_T_sup = desired_T_sups[i]
    desired_Chi_sup = desired_Chi_sups[i]

    # 1) Rotary heat exchanger

    # MOISTURE CONTROL
    # desired_u_rot = (desired_Chi_sup - Chi_out) / ((u_sup/u_ext)*eta_rot_Chi*(Chi_in - Chi_out))

    desired_u_rot = eta_rot_Chi*(u_ext/u_sup)*((Chi_in - Chi_out)/(desired_Chi_sup-Chi_out))
    # print("desired u rot: ", desired_u_rot)
    # print(test2)
    # print()

    u_rot = min(max(desired_u_rot, 0), 1) # Keep u_rot between 0 and 1

    T_rot = (u_ext/u_sup)*u_rot*eta_rot_T*(T_in - T_out) + T_out
    Chi_rot = (u_ext/u_sup)*u_rot*eta_rot_Chi*(Chi_in - Chi_out) + Chi_out

    # 2) Supply fan
    T_fan = T_rot + 1

    # 3) Cooling and dehumidification coil
    Q_cool = 0
    if Chi_rot > desired_Chi_sup:
        # Enthalpy at cooling inlet
        W = Chi_rot/rho_air
        h = HAPropsSI('H','T',T_fan+273.15,'P',101325,'W',W/1000)

        # Latent cooling
        Q_cool_latent = u_sup*rho_air*(Chi_rot - desired_Chi_sup)*Lambda

        # Saturated air temperature at cooling outlet
        T_cool = HAPropsSI('T','H',h,'P',101325,'R',1) - 273.15
        Q_cool_sens = u_sup*rho_air*(T_rot - T_cool)*c_air

        Q_cool = Q_cool_latent + Q_cool_sens

        Chi_cool = desired_Chi_sup

    elif T_rot > desired_T_sup:
        # Sensible cooling
        Q_cool = u_sup*rho_air*(T_rot - desired_T_sup)*c_air

        T_cool = desired_T_sup

        Chi_cool = Chi_rot

    else:
        T_cool = T_fan
        Chi_cool = Chi_rot


    # 4) Heating coil
    Q_heat = 0
    if T_cool < desired_T_sup:
        # Sensible heating
        Q_heat = u_sup*rho_air*(desired_T_sup - T_cool)*c_air

    T_heat = desired_T_sup

    # 5) Humidifier
    Q_humid = 0
    if Chi_cool < desired_Chi_sup:
        desired_W_sup = (desired_Chi_sup/rho_air)/1000 # kg/kg
        W_cool = (Chi_cool/rho_air)/1000 # kg/kg
        Q_humid = u_sup*rho_air*(desired_W_sup - W_cool)*Lambda # kW
        # Q_humid = Q_humid*1000
        Q_humid = Q_humid*1000 * 0.02

    new_dates.append(dates[i])
    new_P_lights.append(P_lights[i])
    u_rots.append(u_rot)
    Q_cools.append(Q_cool)
    Q_heats.append(Q_heat)
    Q_humids.append(Q_humid)



# Create CSV file if it doesn't exist
now_datetime_str = datetime.datetime.now().strftime("%d%m%y-%H%M")
file_name = '../render/csv/' + now_datetime_str + '_hvac_calculations.csv'
dir_name = os.path.dirname(file_name)
os.makedirs(dir_name, exist_ok=True)

# Save results to CSV file
csv_file = open(file_name, 'w', newline='')
writer = csv.writer(csv_file)

# Create the header row
header = ["Date", "u_rot", "Q_cool", "Q_heat", "Q_humid", "P_light"]
writer.writerow(header)

data = []

for i in range(0, int(len(T_ins)/(60*60)) + 1):
    data_row = [new_dates[i], u_rots[i], Q_cools[i], Q_heats[i], Q_humids[i], new_P_lights[i]]

    data.append(data_row)

# Write the data row
writer = csv.writer(csv_file)
writer.writerows(data)
