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


df = pd.read_csv('../render/csv/290524-1305_simulation.csv')

date_strings = df['Date'].to_numpy()
dates = pd.to_datetime(date_strings)

solutions = df[['T_in', 'Chi_in', 'CO2_in', 'T_env', 'T_sup', 'Chi_sup', 'X_ns', 'X_s']].to_numpy().T

# all_data = df[['T_out', 'RH_out', 'Electricity price']].to_numpy().T

solutions = solutions[:, :60*60*24+1:]
dates = dates[:60*60*24+1:]
# all_data = all_data[:, ::60*60]

fig, ax = plt.subplots(1, 1, figsize=(10*0.6, 6*0.6), layout='constrained')

ax.set_ylabel('Temperature [°C]')


ax.plot(dates, solutions[0, :], linewidth=2, alpha=1) # T_in
ax.plot(dates, solutions[3, :], linewidth=2, linestyle='--', alpha=1) # T_env
# ax.plot(dates, all_data[0, :], linewidth=2, c='g', alpha=1) # T_out
ax.axhline(y=16, color='g', linewidth=2, alpha=1) # T_out
ax.legend([r"$T_\mathrm{in}$", r"$T_\mathrm{env}$", r"$T_\mathrm{out}$"])
ax.set_xlabel('Hours')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
ax.xaxis.set_major_locator(mdates.DayLocator())
# plt.setp(ax.get_xticklabels(), rotation=50, ha='right')

# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))

def custom_hour_formatter(x, pos):
    return '{:g}'.format(x.hour)

# ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x) - (19393)}'))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: custom_hour_formatter(mdates.num2date(x), pos)))
# plt.setp(ax.xaxis.get_minorticklabels(), rotation=50, ha='right')

ticks = ax.get_xticks()
ticklabels = [custom_hour_formatter(mdates.num2date(tick), 0) for pos, tick in enumerate(ticks)]
ticklabels[-1] = '24'
ticklabels[0] = '0'

ax.set_xticklabels(ticklabels)

ax.set_xlim(dates[0], dates[-1])
ax.set_ylim(15,24.5)

plt.show()