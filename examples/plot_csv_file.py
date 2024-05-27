import sys
import os
# setting path for running local script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
from render.plot import *

df = pd.read_csv('../render/csv/270524-1549_simulation.csv')

date_strings = df['Date'].to_numpy()
dates = pd.to_datetime(date_strings)

solutions = df[['T_in', 'Chi_in', 'CO2_in', 'T_env', 'T_sup', 'Chi_sup', 'X_ns', 'X_s']].to_numpy().T

climate_attrs = df[['T_hvac', 'Chi_hvac', 'Chi_out', 'CO2_out', 'T_crop']].to_numpy().T

crop_attrs = df[['LAI', 'CAC']].to_numpy().T

actions = df[['u_rot', 'u_fan', 'u_cool', 'u_heat', 'u_humid', 'u_c_inj', 'PPFD']].to_numpy().T

all_data = df[['T_out', 'RH_out', 'Electricity price']].to_numpy().T

Qs = df[['Q_env', 'Q_sens_plant', 'Q_light', 'Q_hvac']].to_numpy().T

Phis = df[['Phi_trans', 'Phi_hvac', 'Phi_c_inj', 'Phi_c_hvac', 'Phi_c_ass']].to_numpy().T

print(Qs.shape)

plot_climate_figure(dates, solutions, climate_attrs, all_data)
plot_crop_figure(dates, solutions, crop_attrs)
# plot_control_input_figure(dates, actions)

# Only plot Qs and Phis for every hour (not second)
new_dates = dates[::3600]
new_Qs = Qs[:, ::3600]
new_Phis = Phis[:, ::3600]

plot_Qs(new_dates, new_Qs)
plot_Phis(new_dates, new_Phis)

plt.waitforbuttonpress()