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

solutions = df[['Phi_c_inj', 'Phi_c_ass', 'Phi_c_hvac']].to_numpy().T

# Only plot every hour to skip variable solve_ivp results
solutions = solutions[:, ::60*60]
dates = dates[::60*60]

CO2_fig, CO2_axes = plt.subplots(3, 1, figsize=(10*0.6, 6*0.6), sharex=True, layout='constrained')
CO2_fig.supxlabel('Days')

CO2_ax1 = CO2_axes[0]
CO2_ax1.plot(dates, solutions[0, :], linewidth=2)
CO2_ax1.legend([r"$\phi_\mathrm{c,inj}$"])
# , r"$\phi_\mathrm{c,ass}$", r"$\phi_\mathrm{c,hvac}$"]) #, loc='upper left')

CO2_ax2 = CO2_axes[1]
CO2_ax2.set_ylabel('$\mathrm{Carbon \ dioxide \ transfer} \ [\mathrm{g} \ \mathrm{s}^{-1}]$')
CO2_ax2.plot(dates, solutions[1, :], linewidth=2, color='orange')
CO2_ax2.legend([r"$\phi_\mathrm{c,ass}$"])

CO2_ax3 = CO2_axes[2]
CO2_ax3.plot(dates, solutions[2, :], linewidth=2, color='green')
CO2_ax3.legend([r"$\phi_\mathrm{c,hvac}$"])

# CO2_ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
CO2_ax3.xaxis.set_major_locator(mdates.DayLocator())
CO2_ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x) - (19393)}'))
# plt.setp(CAC_ax.get_xticklabels(), rotation=50, ha='right')

# Set shared x-axis limits
CO2_ax3.set_xlim(dates[0], dates[-1])

# CO2_fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')

plt.show()
