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

# Import data from CSV file
df = pd.read_csv('../csv/summer_LED80_hvac_calculations.csv')

date_strings = df['Date'].to_numpy()
dates = pd.to_datetime(date_strings)

y = df[['u_rot', 'Q_cool', 'Q_heat', 'Q_humid', 'P_light']].to_numpy().T

fig, axes = plt.subplots(3, 1, figsize=(10*0.6, 6*0.6), sharex=True, layout='constrained')
fig.supxlabel('Days')

ax0 = axes[0]
ax0.fill_between(dates, y[1, :], alpha=0.8)
ax0.set_ylim(bottom=0)
ax0.legend([r"$U_\mathrm{cool}$"], loc='upper left')

ax1 = axes[1]
ax1.fill_between(dates, y[2, :], color='red', alpha=0.8, edgecolor='none')
ax1.set_ylim(bottom=0)
ax1.legend([r"$U_\mathrm{heat}$"], loc='upper left')

ax2 = axes[2]
ax2.fill_between(dates, y[3, :]*5, alpha=0.8, color='green', edgecolor='none')
ax2.set_ylim(bottom=0)
ax2.legend([r"$U_\mathrm{humid}$"])

ax1.set_ylabel('Energy consumption [W]')

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax2.xaxis.set_major_locator(mdates.DayLocator())
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x) - (19393 + 155)}'))

# Set shared x-axis limits
ax2.set_xlim(dates[0], dates[-1])

plt.show(block=True)