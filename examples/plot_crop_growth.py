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

df = pd.read_csv('../render/csv/constant_crop_growth_simulation.csv')

date_strings = df['Date'].to_numpy()
dates = pd.to_datetime(date_strings)

solutions = df[['T_in', 'Chi_in', 'CO2_in', 'T_env', 'T_sup', 'Chi_sup', 'X_ns', 'X_s', 'fresh_weight_shoot_per_plant', 'f_phot']].to_numpy().T

crop_attrs = df[['LAI', 'CAC']].to_numpy().T

actions = df[['PPFD']].to_numpy().T

# FIGURE 1: X_sdw, X_nsdw, fresh shoot_weight_per_plant, PPFD

crop_fig, crop_axes = plt.subplots(2, 1, figsize=(10*0.6, 6*0.6), sharex=True, layout='constrained')
crop_fig.supxlabel('Days')

X_ax = crop_axes[0]
X_ax.set_ylabel('$\mathrm{Dry \ weight} \ [\mathrm{g} \ \mathrm{m}^{-2}]$')
X_ax.plot(dates, solutions[6, :], linewidth=2)
X_ax.plot(dates, solutions[7, :], linewidth=2)
X_ax.legend([r"$X_\mathrm{nsdw}$", r"$X_\mathrm{sdw}$"]) #, loc='upper left')

X_fw_ax = crop_axes[1]
X_fw_ax.set_ylabel('$\mathrm{Fresh \ weight} \ [\mathrm{g} \ \mathrm{plant}^{-1}]$')
X_fw_ax.plot(dates, solutions[8,:], linewidth=2, c='g')
X_fw_ax.legend([r"$X_\mathrm{fw,sht}$"])

# Set shared date format on the x-axis
X_fw_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
X_fw_ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
X_fw_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x) - (19393+155)}'))
# plt.setp(CAC_ax.get_xticklabels(), rotation=50, ha='right')

# Set shared x-axis limits
X_fw_ax.set_xlim(dates[0], dates[-1])

# plt.savefig("../examples/simulations/crop_growth.pgf")





# Second figure with LAI, CAC, f_phot (as the crop grows thesse evolve --> parameters that the indoor climate model is dependent on)

LAI_fig, LAI_axes = plt.subplots(2, 1, figsize=(10*0.6, 6*0.6), sharex=True, layout='constrained')
LAI_fig.supxlabel('Days')

LAI_ax = LAI_axes[0]
LAI_ax.set_ylabel('$\mathrm{Leaf \ area \ index} \ [\mathrm{-}]$')
LAI_ax.plot(dates, crop_attrs[0,:], linewidth=2)
LAI_ax.legend([r"$\mathit{LAI}$"])

f_phot_ax = LAI_axes[1]
f_phot_ax.set_ylabel('$\mathrm{Photosynthesis} \ [\mathrm{g} \ \mathrm{m}^{-2} \ \mathrm{s}^{-1}]$')
f_phot_ax.plot(dates, solutions[9,:], linewidth=2, c='g')
f_phot_ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
f_phot_ax.legend([r"$f_\mathrm{phot}$"])

# Set shared date format on the x-axis
f_phot_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
f_phot_ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
f_phot_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x) - (19393+155)}'))
# plt.setp(CAC_ax.get_xticklabels(), rotation=50, ha='right')

# Set shared x-axis limits
f_phot_ax.set_xlim(dates[0], dates[-1])

plt.show()




