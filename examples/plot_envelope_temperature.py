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
})

# Default style settings
plt.rcParams.update({
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
})


df = pd.read_csv('../render/csv/envelope_temperature_simulation.csv')

date_strings = df['Date'].to_numpy()
dates = pd.to_datetime(date_strings)

solutions = df[['T_in', 'Chi_in', 'CO2_in', 'T_env', 'T_sup', 'Chi_sup', 'X_ns', 'X_s']].to_numpy().T

all_data = df[['T_out', 'RH_out', 'Electricity price']].to_numpy().T

fig, ax = plt.subplots(1, 1, figsize=(10, 10), layout='constrained')

ax.set_ylabel('Temperature [°C]')
ax.plot(dates, solutions[0, :], linewidth=2, alpha=1) # T_in
ax.plot(dates, solutions[3, :], linewidth=1) # T_env
ax.plot(dates, all_data[0, :], linestyle='--', linewidth=1) # T_out
ax.legend([r"$T_\mathrm{in}$", r"$T_\mathrm{env}$", r"$T_\mathrm{out}$"])
ax.set_xlabel('Date')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax.xaxis.set_major_locator(mdates.DayLocator())
plt.setp(ax.get_xticklabels(), rotation=50, ha='right')

ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
plt.setp(ax.xaxis.get_minorticklabels(), rotation=50, ha='right')

ax.set_xlim(dates[0], dates[-1])

plt.show()