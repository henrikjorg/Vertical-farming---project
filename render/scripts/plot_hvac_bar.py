import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,

    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

df = pd.read_csv('../csv/winter_LED50_hvac_calculations.csv')
date_strings = df['Date'].to_numpy()
dates = pd.to_datetime(date_strings)

y = df[['u_rot', 'Q_cool', 'Q_heat', 'Q_humid', 'P_light']].to_numpy().T
Q_cool = sum(y[1, :])/1000
Q_heat = sum(y[2, :])/1000
Q_humid = 5*sum(y[3, :])/1000
P_light = sum(y[4, :])/1000

df2 = pd.read_csv('../csv/summer_LED50_hvac_calculations.csv')
y2 = df2[['u_rot', 'Q_cool', 'Q_heat', 'Q_humid', 'P_light']].to_numpy().T
Q_cool2 = sum(y2[1, :])/1000
Q_heat2 = sum(y2[2, :])/1000
Q_humid2 = 5*sum(y2[3, :])/1000
P_light2 = sum(y2[4, :])/1000

df3 = pd.read_csv('../csv/winter_LED80_hvac_calculations.csv')
y3 = df3[['u_rot', 'Q_cool', 'Q_heat', 'Q_humid', 'P_light']].to_numpy().T
Q_cool3 = sum(y3[1, :])/1000
Q_heat3 = sum(y3[2, :])/1000
Q_humid3 = 5*sum(y3[3, :])/1000
P_light3 = sum(y3[4, :])/1000

df4 = pd.read_csv('../csv/summer_LED80_hvac_calculations.csv')
y4 = df4[['u_rot', 'Q_cool', 'Q_heat', 'Q_humid', 'P_light']].to_numpy().T
Q_cool4 = sum(y4[1, :])/1000
Q_heat4 = sum(y4[2, :])/1000
Q_humid4 = 5*sum(y4[3, :])/1000
P_light4 = sum(y4[4, :])/1000

# Make arrays for plotting bars
P_lights = [P_light, P_light2, P_light3, P_light4]
Q_cools = [Q_cool, Q_cool2, Q_cool3, Q_cool4]
Q_heats = [Q_heat, Q_heat2, Q_heat3, Q_heat4]
Q_humids = [Q_humid, Q_humid2, Q_humid3, Q_humid4]

sim1_sum = P_light + Q_cool + Q_heat + Q_humid
sim2_sum = P_light2 + Q_cool2 + Q_heat2 + Q_humid2
sim3_sum = P_light3 + Q_cool3 + Q_heat3 + Q_humid3
sim4_sum = P_light4 + Q_cool4 + Q_heat4 + Q_humid4
sums = [sim1_sum, sim2_sum, sim3_sum, sim4_sum]

# Make bar plot
fig, ax = plt.subplots(figsize=(10*0.6, 6*0.6), layout='constrained')
# ax.grid(True, zorder=0, color = "lightgrey", ls = "--", alpha=0.5, axis='y')

barWidth = 0.7
index = range(4)

index = [1, 2, 4, 5]

ax.bar(index, P_lights, width=barWidth, color='tab:blue', label=r"$Q_\mathrm{light}$", alpha=0.8, zorder=3)
ax.bar(index, Q_heats, bottom=P_lights, color='tab:orange', width=barWidth, label=r"$Q_\mathrm{heat}$", alpha=0.8, zorder=3)
ax.bar(index, Q_cools, bottom=[x + y for x, y in zip(P_lights, Q_heats)], color='tab:green',  width=barWidth, label=r"$Q_\mathrm{cool}$", alpha=0.8, zorder=3)
ax.bar(index, Q_humids, bottom=[x + y + z for x, y, z in zip(P_lights, Q_cools, Q_heats)], color='tab:red', width=barWidth, label=r"$Q_\mathrm{humid}$", alpha=0.8, zorder=3)

# Annotate total energy consumption
# xs = [1, 2, 4, 5]
# for i, y in enumerate(sums):
#     ax.annotate(
#         xy = (xs[i], y),
#         text = f"{y:.0f} kWh",
#         xytext = (0, 5),
#         size=8,
#         # color='gray',
#         alpha=0.8,
#         textcoords = "offset points",
#         ha = "center",
#         va = "center",
#         weight = "bold"
#     )

ax.text(1.05, 1700, r"$\eta_\mathrm{light}$ = $50$ \%", rotation=0, verticalalignment='center', color='black')
ax.text(4.1, 1260, r"$\eta_\mathrm{light}$ = $80$ \%", rotation=0, verticalalignment='center', color='black')

plt.xticks(index, ['Winter', 'Summer', 'Winter', 'Summer'])
plt.ylabel('Energy consumption [kWh]')

ax.set_xlim(0, 6)
ax.set_ylim(0, 2000)

plt.legend()

plt.show()