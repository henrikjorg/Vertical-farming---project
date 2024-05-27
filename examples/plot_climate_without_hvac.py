import sys
import os
# setting path for running local script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,

    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


df = pd.read_csv('../render/csv/5_days_without_hvac_simulation.csv')

date_strings = df['Date'].to_numpy()
dates = pd.to_datetime(date_strings)

solutions = df[['T_in', 'Chi_in', 'CO2_in', 'T_env', 'T_sup', 'Chi_sup', 'X_ns', 'X_s']].to_numpy().T

all_data = df[['T_out', 'RH_out', 'Electricity price']].to_numpy().T

fig, axes = plt.subplots(2, 1, figsize=(10*0.6, 6*0.6), sharex=True, layout='constrained')

temp_ax = axes[0]
temp_ax.set_ylabel('Temperature [°C]')

temp_ax.plot(dates, solutions[0, :], linewidth=2, alpha=1) # T_in
temp_ax.text(dates[-6], 24, 'Desired temperature', rotation=0, verticalalignment='center', color='gray', alpha=0.8)
temp_ax.axhline(y=solutions[0, 0], color='gray', linestyle='--', linewidth=1, alpha=0.8) # T_des
temp_ax.legend([r"$T_\mathrm{in}$"])


humid_ax = axes[1]
humid_ax.set_ylabel('$\mathrm{Absolute \ humidity} \ [\mathrm{g} \ \mathrm{m}^{-3}]$')
humid_ax.plot(dates, solutions[1, :], linewidth=2, alpha=1) # Chi_in
humid_ax.axhline(y=solutions[1, 0], color='gray', linestyle='--', linewidth=1, alpha=0.8) # T_des

humid_ax.set_xlabel('Date')

humid_ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
humid_ax.xaxis.set_major_locator(mdates.DayLocator())
plt.setp(humid_ax.get_xticklabels(), rotation=50, ha='right')

humid_ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
humid_ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
plt.setp(humid_ax.xaxis.get_minorticklabels(), rotation=50, ha='right')

humid_ax.set_xlim(dates[0], dates[-1])

plt.show()