import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# UNCOMMENT TO: Configure LaTex style for master thesis rendering
# plt.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'serif',
#     'font.serif': ['Computer Modern Roman'],
# })

# Default style settings
plt.rcParams.update({
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

"""Functions to plot data from a csv file"""
def plot_climate_figure(dates, y, climate_attrs, all_data):
    climate_fig, climate_axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, dpi=100)
    climate_fig.supxlabel('Date')

    temp_ax = climate_axes[0]
    temp_ax.set_ylabel('Temperature [°C]')
    temp_ax.plot(dates, y[0, :], linewidth=2) # T_in
    temp_ax.plot(dates, y[3, :], linewidth=1) # T_env
    # temp_ax.plot(dates, y[4, :], linewidth=1) # T_sup
    temp_ax.plot(dates, all_data[0, :], linestyle='-', linewidth=1)[0] # T_out
    # temp_ax.plot(dates, climate_attrs[4, :], linewidth=1)[0] # T_crop
    temp_ax.axhline(y=y[0, 0], color='grey', linestyle='--', linewidth=1.5, alpha=0.5) # T_des
    temp_ax.legend(["T_in", "T_env", "T_out", "T_des"])
    # temp_ax.legend(["T_in", "T_env", "T_sup", "T_out", "T_des"])

    humid_ax = climate_axes[1]
    humid_ax.set_ylabel('Humidity [g/m³]')
    humid_ax.plot(dates, y[1, :], linewidth=2)[0]
    humid_ax.plot(dates, climate_attrs[2, :], linestyle='-', linewidth=1)[0]
    humid_ax.plot(dates, y[5, :], linewidth=2)[0]
    # humid_ax.plot(dates, climate_attrs[1, :], linewidth=1)[0]
    humid_ax.axhline(y=y[1, 0], color='grey', linestyle='--', linewidth=1.5, alpha=0.5)
    humid_ax.legend(["Chi_in", "Chi_out", "Chi_sup", "Chi_des"])

    co2_ax = climate_axes[2]
    co2_ax.set_ylabel('CO2 [ppm]')
    co2_ax.plot(dates, y[2, :], linewidth=2)[0]
    co2_ax.plot(dates, climate_attrs[3, :], linestyle='-', linewidth=1)[0]
    co2_ax.axhline(y=y[2, 0], color='grey', linestyle='--', linewidth=1.5, alpha=0.5)
    co2_ax.legend(["CO2_in", "CO2_out", "CO2_des"])

    # Set shared date format on the x-axis
    co2_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Set date format
    co2_ax.xaxis.set_major_locator(mdates.DayLocator(interval=1)) # Set interval to 1 day
    plt.setp(co2_ax.get_xticklabels(), rotation=45, ha='right') # Rotate labels for better visibility

    # Set shared x-axis limits
    co2_ax.set_xlim(dates[0], dates[-1])

    plt.show(block=False)


def plot_crop_figure(dates, y, crop_attrs):
    crop_fig, crop_axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, dpi=100)
    crop_fig.supxlabel('Date')

    X_ns_ax = crop_axes[0]
    X_ns_ax.set_ylabel('X_ns [g/m²]')
    X_ns_ax.plot(dates, y[6, :], linewidth=2)

    X_s_ax = crop_axes[1]
    X_s_ax.set_ylabel('X_s [g/m²]')
    X_s_ax.plot(dates, y[7, :], linewidth=2)

    LAI_ax = crop_axes[2]
    LAI_ax.set_ylabel('LAI [m²/m²]')
    LAI_ax.plot(dates, crop_attrs[0,:], linewidth=2)

    # Set shared date format on the x-axis
    LAI_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    LAI_ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(LAI_ax.get_xticklabels(), rotation=45, ha='right')

    # Set shared x-axis limits
    LAI_ax.set_xlim(dates[0], dates[-1])

    plt.show(block=False)

def plot_control_input_figure(dates, actions):
    u_fig, u_axes = plt.subplots(7, 1, figsize=(10, 10), sharex=True, dpi=100)
    u_fig.supxlabel('Date')

    labels = ['u_rot', 'u_fan', 'u_cool', 'u_heat', 'u_humid', 'u_c_inj', 'u_light']
    for i, ax in enumerate(u_axes):
            ax.set_ylabel(labels[i])
            ax.plot(dates, actions[i,:], linewidth=2)

    # Set shared date format on the x-axis
    u_axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    u_axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(u_axes[-1].get_xticklabels(), rotation=45, ha='right')

    # Set shared x-axis limits
    u_axes[-1].set_xlim(dates[0], dates[-1])

    plt.show(block=False)