# PLOTTE!
# 1) u_rot 2) P_light, Q_cool, Q_heat, Q_humid

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# UNCOMMENT TO: Configure LaTex style for master thesis rendering
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

# Import data from CSV file
df = pd.read_csv('../render/csv/250524-1523_hvac_calculations.csv')

date_strings = df['Date'].to_numpy()
dates = pd.to_datetime(date_strings)

y = df[['u_rot', 'Q_cool', 'Q_heat', 'Q_humid', 'P_light']].to_numpy().T

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, layout='constrained')
fig.supxlabel('Date')

ax1 = axes[0]
ax1.set_ylabel('Thermal power consumption [W]')
ax1.fill_between(dates, y[1, :], alpha=0.5)
ax1.fill_between(dates, y[2, :], alpha=0.5)
ax1.fill_between(dates, y[3, :], alpha=0.5)
ax1.set_ylim(bottom=0)
ax1.legend([r"$Q_\mathrm{cool}$", r"$Q_\mathrm{heat}$", r"$Q_\mathrm{humid}$"])

ax2 = axes[1]
ax2.set_ylabel('Rotary heat exchanger control [0-1]')
ax2.fill_between(dates, y[0, :], alpha=0.5)
ax2.set_ylim(bottom=0)

ax3 = axes[2]
ax3.set_ylabel('Light power consumption [W]')
ax3.fill_between(dates, y[4, :], alpha=0.5)
ax3.set_ylim(bottom=0)

# Set shared date format on the x-axis
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Set date format
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=1)) # Set interval to 1 day
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right') # Rotate labels for better visibility

# Set shared x-axis limits
ax3.set_xlim(dates[0], dates[-1])

plt.show(block=True)