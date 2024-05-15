import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime
import os

# Configuration
PLOT_CONFIG = {
   'render_climate': True,
   'render_crop': False,
   'render_control_input': False,
}

# # UNCOMMENT TO: Configure LaTex style for master thesis rendering
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

class RenderLive:
    def __init__(self, config, start_date, t, y):
        self.t = t
        self.y = y
        self.start_date = start_date
        self.dates = [self.start_date + datetime.timedelta(seconds=i) for i in t]
        self.empty_y = [np.nan] * len(self.dates)

        # Initialize figures only if needed
        if PLOT_CONFIG['render_climate']:
            self._init_climate_figure()
        if PLOT_CONFIG['render_crop']:
            self._init_crop_figure()
        if PLOT_CONFIG['render_control_input']:
            self._init_control_input_figure()

    def _init_climate_figure(self):
        self.climate_fig, self.climate_axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, dpi=100)
        self.climate_fig.supxlabel('Date')

        # Initialize temperature subplot
        self.temp_ax = self.climate_axes[0]
        self.temp_ax.set_ylabel('Temperature [°C]')

        self.T_in_line = self.temp_ax.plot(self.dates, self.empty_y, linewidth=2)[0]
        self.T_out_line = self.temp_ax.plot(self.dates, self.empty_y, linestyle='-', linewidth=1)[0]
        self.T_sup_line = self.temp_ax.plot(self.dates, self.empty_y, linewidth=1)[0]
        self.temp_ax.axhline(y=self.y[0, 0], color='grey', linestyle='--', linewidth=1.5, alpha=0.5)
        self.temp_ax.legend(["T_in", "T_out", "T_hvac", "T_des"])

        # Humidity subplot
        self.humid_ax = self.climate_axes[1]
        self.humid_ax.set_ylabel('Humidity [g/m³]')

        self.Chi_in_line = self.humid_ax.plot(self.dates, self.empty_y, linewidth=2)[0]
        self.Chi_out_line = self.humid_ax.plot(self.dates, self.empty_y, linestyle='-', linewidth=1)[0]
        self.Chi_sup_line = self.humid_ax.plot(self.dates, self.empty_y, linewidth=1)[0]
        self.humid_ax.axhline(y=self.y[1, 0], color='grey', linestyle='--', linewidth=1.5, alpha=0.5)
        self.humid_ax.legend(["Chi_in", "Chi_out", "Chi_hvac", "Chi_des"])

        # CO2 subplot
        self.co2_ax = self.climate_axes[2]
        self.co2_ax.set_ylabel('CO2 [ppm]')

        self.CO2_in_line = self.co2_ax.plot(self.dates, self.empty_y, linewidth=2)[0]
        self.CO2_out_line = self.co2_ax.plot(self.dates, self.empty_y, linestyle='-', linewidth=1)[0]
        self.co2_ax.axhline(y=self.y[2, 0], color='grey', linestyle='--', linewidth=1.5, alpha=0.5)
        self.co2_ax.legend(["CO2_in", "CO2_out", "CO2_des"])

        # Set shared date format on the x-axis
        self.co2_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Set date format
        self.co2_ax.xaxis.set_major_locator(mdates.DayLocator(interval=1)) # Set interval to 1 day
        plt.setp(self.co2_ax.get_xticklabels(), rotation=45, ha='right') # Rotate labels for better visibility

    def _init_crop_figure(self):
        self.crop_fig, self.crop_axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, dpi=100)
        self.crop_fig.supxlabel('Date')

        self.X_ns_ax = self.crop_axes[0]
        self.X_ns_ax.set_ylabel('X_ns [g/m²]')
        self.X_ns_line = self.X_ns_ax.plot(self.dates, self.empty_y, linewidth=2)[0]

        self.X_s_ax = self.crop_axes[1]
        self.X_s_ax.set_ylabel('X_s [g/m²]')
        self.X_s_line = self.X_s_ax.plot(self.dates, self.empty_y, linewidth=2)[0]

        self.LAI_ax = self.crop_axes[2]
        self.LAI_ax.set_ylabel('LAI [m²/m²]')
        self.LAI_line = self.LAI_ax.plot(self.dates, self.empty_y, linewidth=2)[0]

        # Set shared date format on the x-axis
        self.LAI_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.LAI_ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(self.LAI_ax.get_xticklabels(), rotation=45, ha='right')

    def _init_control_input_figure(self):
        self.u_fig, self.u_axes = plt.subplots(7, 1, figsize=(10, 10), sharex=True, dpi=100)
        self.u_fig.supxlabel('Date')

        labels = ['u_rot', 'u_fan', 'u_cool', 'u_heat', 'u_humid', 'u_c_inj', 'u_light']
        self.u_lines = []
        for i, ax in enumerate(self.u_axes):
            if i < 6: # u_light must be accessable separately
                ax.set_ylabel(labels[i])
                self.u_lines.append(ax.plot(self.dates, self.empty_y, linewidth=2)[0])

        self.u_light_ax = self.u_axes[-1]
        self.u_light_line = self.u_axes[-1].plot(self.dates, self.empty_y, linewidth=2)[0]

        # Set shared date format on the x-axis
        self.u_axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.u_axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(self.u_axes[-1].get_xticklabels(), rotation=45, ha='right')

    def _render_climate(self, x_lim, y, climate_attrs, all_data):
        # Update temperature lines
        self.T_in_line.set_ydata(y[0, :])
        self.T_out_line.set_ydata(all_data[0, :])
        self.T_sup_line.set_ydata(climate_attrs['T_hvac'])

        # Update temperature y-axis limits
        combined_y = np.concatenate([y[0, :x_lim], all_data[0, :x_lim], climate_attrs['T_hvac'][:x_lim]])
        y_min, y_max = combined_y.min(), combined_y.max()
        y_range = y_max - y_min
        margin = 0.1 * y_range
        self.temp_ax.set_ylim(y_min - margin, y_max + margin)

        # Update humidity lines
        self.Chi_in_line.set_ydata(y[1, :])
        self.Chi_out_line.set_ydata(climate_attrs['Chi_out'])
        self.Chi_sup_line.set_ydata(climate_attrs['Chi_hvac'])

        # Update humidity y-axis limits
        combined_y = np.concatenate([y[1, :x_lim], climate_attrs['Chi_out'][:x_lim], climate_attrs['Chi_hvac'][:x_lim]])
        y_min, y_max = combined_y.min(), combined_y.max()
        y_range = y_max - y_min
        margin = 0.1 * y_range
        self.humid_ax.set_ylim(y_min - margin, y_max + margin)

        # Update CO2 lines
        self.CO2_in_line.set_ydata(y[2, :])
        self.CO2_out_line.set_ydata(climate_attrs['CO2_out'])

        # Update CO2 y-axis limits
        combined_y = np.concatenate([y[2, :x_lim], climate_attrs['CO2_out'][:x_lim]])
        y_min, y_max = combined_y.min(), combined_y.max()
        y_range = y_max - y_min
        margin = 0.1 * y_range
        self.co2_ax.set_ylim(y_min - margin, y_max + margin)

        # Update shared x-axis limits
        x_lim_date = self.start_date + datetime.timedelta(seconds=x_lim)
        self.temp_ax.set_xlim(self.start_date, x_lim_date)

    def _render_crop(self, x_lim, y, crop_attrs):
        self.X_ns_line.set_ydata(y[3, :])
        y_min, y_max = y[3, :x_lim].min(), y[3, :x_lim].max()
        margin = 0.1 * (y_max - y_min)
        self.X_ns_ax.set_ylim(y_min - margin, y_max + margin)

        self.X_s_line.set_ydata(y[4, :])
        y_min, y_max = y[4, :x_lim].min(), y[4, :x_lim].max()
        margin = 0.1 * (y_max - y_min)
        self.X_s_ax.set_ylim(y_min - margin, y_max + margin)

        self.LAI_line.set_ydata(crop_attrs['LAI'])
        y_min, y_max = crop_attrs['LAI'][:x_lim].min(), crop_attrs['LAI'][:x_lim].max()
        margin = 0.1 * (y_max - y_min)
        self.LAI_ax.set_ylim(y_min - margin, y_max + margin)

        # Update shared x-axis limits
        x_lim_date = self.start_date + datetime.timedelta(seconds=x_lim)
        self.X_ns_ax.set_xlim(self.start_date, x_lim_date)

    def _render_control_input(self, x_lim, actions, all_data):
        for i, (ax, line) in enumerate(zip(self.u_axes, self.u_lines)):
            if i < 6:
                line.set_ydata(actions[i, :])
                y_min, y_max = actions[i, :x_lim].min(), actions[i, :x_lim].max()
                margin = 0.1 * (y_max - y_min)
                ax.set_ylim(y_min - margin, y_max + margin)

        # Update u_light separately
        self.u_light_line.set_ydata(all_data[3, :])
        y_min, y_max = all_data[3, :x_lim].min(), all_data[3, :x_lim].max()
        margin = 0.1 * (y_max - y_min)
        self.u_light_ax.set_ylim(y_min - margin, y_max + margin)

        # Update shared x-axis limits
        x_lim_date = self.start_date + datetime.timedelta(seconds=x_lim)
        self.u_light_ax.set_xlim(self.start_date, x_lim_date)

    def render(self, terminated, current_step, t, model, solutions, climate_attrs, crop_attrs, actions, all_data, index):
        x_lim = int(t[index - 2])  # -2 to prevent showing zero values
        self._render_config(PLOT_CONFIG, x_lim, solutions, climate_attrs, crop_attrs, actions, all_data)

        if terminated:
            x_lim = int(t[-1])
            self._render_config(PLOT_CONFIG, x_lim, solutions, climate_attrs, crop_attrs, actions, all_data)
            plt.waitforbuttonpress()
            return
    
        plt.pause(0.0001)

    def _render_config(self, plot_config, x_lim, solutions, climate_attrs, crop_attrs, actions, all_data):
        if plot_config['render_climate']:
            self._render_climate(x_lim, solutions, climate_attrs, all_data)
        if plot_config['render_crop']:
            self._render_crop(x_lim, solutions, crop_attrs)
        if plot_config['render_control_input']:
            self._render_control_input(x_lim, actions, all_data)

    def close(self):
        # Save the figures
        datetime_str = datetime.datetime.now().strftime("%d%m%y-%H%M")
        if PLOT_CONFIG['render_climate']:
            self._save_figure(self.climate_fig, 'climate', datetime_str)
        if PLOT_CONFIG['render_crop']:
            self._save_figure(self.crop_fig, 'crop', datetime_str)
        if PLOT_CONFIG['render_control_input']:
            self._save_figure(self.u_fig, 'control_input', datetime_str)
        
        plt.close()

    def _save_figure(self, figure, name, datetime_str):
        """Utility function to save figures."""
        folder = '../render/plots/'
        filename = f"{datetime_str}_{name}.png"
        os.makedirs(folder, exist_ok=True) # Create the directory if it doesn't exist
        figure.savefig(folder + filename, dpi=100, bbox_inches='tight')
