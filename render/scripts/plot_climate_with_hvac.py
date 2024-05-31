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


df = pd.read_csv('../csv/summer_LED80_simulation.csv')

date_strings = df['Date'].to_numpy()
dates = pd.to_datetime(date_strings)

solutions = df[['T_in', 'Chi_in', 'T_sup', 'Chi_sup']].to_numpy().T
all_data = df[['T_out', 'Chi_out']].to_numpy().T

# Only plot every hour to skip variable solve_ivp results
solutions = solutions[:, ::60*60]
dates = dates[::60*60]
all_data = all_data[:, ::60*60]

fig, axes = plt.subplots(2, 1, figsize=(10*0.6, 6*0.6), sharex=True, layout='constrained')

temp_ax = axes[0]
temp_ax.set_ylabel('Temperature [°C]')

temp_ax.plot(dates, solutions[0, :], linewidth=2, alpha=0.8) # T_in
temp_ax.plot(dates, solutions[2, :], linewidth=2, alpha=1) # T_sup
temp_ax.plot(dates, all_data[0, :], linewidth=2, alpha=0.8) # T_out
temp_ax.legend([r"$T_\mathrm{in}$", r"$T_\mathrm{sup}$", r"$T_\mathrm{out}$"], loc ='lower right')


humid_ax = axes[1]
humid_ax.set_ylabel('$\mathrm{Absolute \ humidity} \ [\mathrm{g} \ \mathrm{m}^{-3}]$')
humid_ax.plot(dates, solutions[1, :], linewidth=2, alpha=0.8) # Chi_in
humid_ax.plot(dates, solutions[3, :], linewidth=2, alpha=1) # Chi_sup
humid_ax.plot(dates, all_data[1, :], linewidth=2, alpha=0.8) # Chi_out
humid_ax.legend([r"$\chi_\mathrm{in}$", r"$\chi_\mathrm{sup}$", r"$\chi_\mathrm{out}$"], loc ='lower right')

humid_ax.set_xlabel('Days')

humid_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
humid_ax.xaxis.set_major_locator(mdates.DayLocator())
humid_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x) - (19393 + 155)}'))


humid_ax.set_xlim(dates[0], dates[-1])

plt.show()