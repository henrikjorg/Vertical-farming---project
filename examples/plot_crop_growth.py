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

plt.rcParams.update({
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 300,
})

df = pd.read_csv('../render/csv/constant_climate_simulation.csv')

date_strings = df['Date'].to_numpy()
dates = pd.to_datetime(date_strings)

solutions = df[['T_in', 'Chi_in', 'CO2_in', 'T_env', 'T_sup', 'Chi_sup', 'X_ns', 'X_s']].to_numpy().T

crop_attrs = df[['LAI', 'CAC']].to_numpy().T

actions = df[['PPFD']].to_numpy().T

crop_fig, crop_axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, layout='constrained')
crop_fig.supxlabel('Days since start of cultivation cycle')

X_ns_ax = crop_axes[0]
X_ns_ax.set_ylabel('$X_\mathrm{nsdw} \ [\mathrm{g} \ \mathrm{m}^{-2}]$')
X_ns_ax.plot(dates, solutions[6, :], linewidth=2)
X_ns_ax.legend([r"$X_\mathrm{nsdw}$"], loc='upper left')

# Create a second y-axis for the PPFD
PPFD_ax = X_ns_ax.twinx()
PPFD_ax.set_ylabel('$[\mathrm{mol} \ \mathrm{m}^{-2} \ \mathrm{s}^{-1}]$')
PPFD_ax.plot(dates, actions[0,:], linewidth=1, alpha=0.5, color='red')
PPFD_ax.legend([r"$\mathit{PPFD}$"])

X_s_ax = crop_axes[1]
X_s_ax.set_ylabel('$X_\mathrm{sdw} \ [\mathrm{g} \ \mathrm{m}^{-2}]$')
X_s_ax.plot(dates, solutions[7, :], linewidth=2, color='orange')
X_s_ax.legend([r"$X_\mathrm{sdw}$"])

CAC_ax = crop_axes[2]
CAC_ax.set_ylabel('$\mathit{CAC} \ [\%]$')
CAC_ax.plot(dates, crop_attrs[1,:]*100, linewidth=2, color='green')

# PPFD_ax = crop_axes[3]
# PPFD_ax.set_ylabel('$\mathit{PPFD} \ [\mathrm{mol} \mathrm{m}^{-2} \mathrm{s}^{-1}]$')
# PPFD_ax.plot(dates, actions[0,:], linewidth=2, color='red')

# Set shared date format on the x-axis
CAC_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
CAC_ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
CAC_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x) - 19393}'))
# plt.setp(CAC_ax.get_xticklabels(), rotation=50, ha='right')

# Set shared x-axis limits
CAC_ax.set_xlim(dates[0], dates[-1])

# plt.savefig("../examples/simulations/crop_growth.pgf")

plt.show()




