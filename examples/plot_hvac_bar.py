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
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
})

folder_name = "sens-CAC_trans-CAC"
# folder_name = "sens-LAI_trans-LAI"

# Import data from CSV file
# df = pd.read_csv('../render/csv/' + folder_name + '/winter_LED81_hvac_calculations.csv')
df = pd.read_csv('../render/csv/270524-1449_hvac_calculations.csv')
date_strings = df['Date'].to_numpy()
dates = pd.to_datetime(date_strings)

y = df[['u_rot', 'Q_cool', 'Q_heat', 'Q_humid', 'P_light']].to_numpy().T
Q_cool = sum(y[1, :])/1000
Q_heat = sum(y[2, :])/1000
Q_humid = sum(y[3, :])/1000
print("P light total consumption 1: ", sum(y[4, :])/1000)

df2 = pd.read_csv('../render/csv/winter_LED52_hvac_calculations.csv')
# df2 = pd.read_csv('../render/csv/250524-1701_hvac_calculations.csv')
y2 = df2[['u_rot', 'Q_cool', 'Q_heat', 'Q_humid', 'P_light']].to_numpy().T
Q_cool2 = sum(y2[1, :])/1000
Q_heat2 = sum(y2[2, :])/1000
Q_humid2 = sum(y2[3, :])/1000
print("P light total consumption 2: ", sum(y2[4, :])/1000)

df3 = pd.read_csv('../render/csv/summer_LED81_hvac_calculations.csv')
y3 = df3[['u_rot', 'Q_cool', 'Q_heat', 'Q_humid', 'P_light']].to_numpy().T
Q_cool3 = sum(y3[1, :])/1000
Q_heat3 = sum(y3[2, :])/1000
Q_humid3 = sum(y3[3, :])/1000
print("P light total consumption 3: ", sum(y3[4, :])/1000)

df4 = pd.read_csv('../render/csv/summer_LED52_hvac_calculations.csv')
y4 = df4[['u_rot', 'Q_cool', 'Q_heat', 'Q_humid', 'P_light']].to_numpy().T
Q_cool4 = sum(y4[1, :])/1000
Q_heat4 = sum(y4[2, :])/1000
Q_humid4 = sum(y4[3, :])/1000
print("P light total consumption 4: ", sum(y4[4, :])/1000)


# Make arrays for plotting bars
Q_cools = [Q_cool, Q_cool2, Q_cool3, Q_cool4]
Q_heats = [Q_heat, Q_heat2, Q_heat3, Q_heat4]
Q_humids = [Q_humid, Q_humid2, Q_humid3, Q_humid4]

sim1_sum = Q_cool + Q_heat + Q_humid
sim2_sum = Q_cool2 + Q_heat2 + Q_humid2
sim3_sum = Q_cool3 + Q_heat3 + Q_humid3
sim4_sum = Q_cool4 + Q_heat4 + Q_humid4
sums = [sim1_sum, sim2_sum, sim3_sum, sim4_sum]

# Make bar plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.grid(True, zorder=0, color = "lightgrey", ls = "--", alpha=0.5, axis='y')

barWidth = 0.5
index = range(4)

ax.bar(index, Q_cools, color='b', width=barWidth, edgecolor='grey', label=r"$Q_\mathrm{cool}$", zorder=3)
ax.bar(index, Q_heats, bottom=Q_cools, color='r', width=barWidth, edgecolor='grey', label=r"$Q_\mathrm{heat}$", zorder=3)
# ax.bar(index, Q_humids, bottom=[x + y for x, y in zip(Q_cools, Q_heats)], color='g', width=barWidth, edgecolor='grey', label=r"$Q_\mathrm{humid}$", zorder=3)

# Annotate total energy consumption
for i, y in enumerate(sums):
    ax.annotate(
        xy = (i, y),
        text = f"{y:.0f} kWh",
        xytext = (0, 7),
        size=8,
        textcoords = "offset points",
        ha = "center",
        va = "center",
        weight = "bold"
    )

plt.xlabel('Scenario')
plt.ylabel('Thermal energy consumption [kWh]')
plt.xticks(index, ['1', '2', '3', '4'])
plt.legend()

plt.show()