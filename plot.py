import matplotlib.pyplot as plt
from config import *
from matplotlib.ticker import MaxNLocator
import numpy as np

# (TODO): Plot desired parameter values (desired temperature etc.)

class Plotter:
    def __init__(self, t, y, crop_attrs = None):
        plt.ion()
        # Convert time from seconds to days for the initial plot setup
        days = t / (24 * 3600)  # Convert seconds to days

        # Environment figure
        self.env_fig, self.env_axes = plt.subplots(len(env_states_info), 1, figsize=(10, len(env_states_info)*3))
        plt.subplots_adjust(hspace=0.5)
        self.env_fig.suptitle('Environment')
        self.lines = []
        for i, (ax, (_, info)) in enumerate(zip(self.env_axes, env_states_info.items())):
            line = ax.plot(days, y[i, :], info.get('color', 'b'))[0]
            ax.set_title(info.get('title', ''))
            ax.set_xlabel('Time [days]')  # Changed to days
            ax.set_ylabel(info.get('title', '-') + ' [' + info['unit'] + ']')
            ax.set_ylim(0, 100)
            ax.set_xlim(days.min(), days.max())  # Adjust limits to days
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure integer values on x-axis
            ax.grid(True)
            
            self.lines.append(line)

        # Crop figure
        self.crop_fig, self.crop_axes = plt.subplots(len(crop_states_info), 1, figsize=(10, len(crop_states_info)*3))
        plt.subplots_adjust(hspace=0.5)
        self.crop_fig.suptitle('Crop')
        for i, (ax, (_, info)) in enumerate(zip(self.crop_axes, crop_states_info.items())):
            line = ax.plot(days, y[i, :], info.get('color', 'b'))[0]
            ax.set_title(info.get('title', ''))
            ax.set_xlabel('Time [days]')  # Changed to days
            ax.set_ylabel(info.get('title', '-') + ' [' + info['unit'] + ']')
            ax.set_ylim(0, 100)
            ax.set_xlim(days.min(), days.max())  # Adjust limits to days
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure integer values on x-axis
            ax.grid(True)
            
            self.lines.append(line)

        self.crop_attr_figs = {}
        self.crop_attr_axes = {}
        self.crop_attr_lines = {}
        if crop_attrs is not None:
            for attr_name, data in crop_attrs.items():
                fig, ax = plt.subplots(1, 1, figsize=(10, 3))
                line, = ax.plot(days, data, label=attr_name)
                ax.set_title(f'{attr_name} over Time')
                ax.set_xlabel('Time [days]')
                ax.set_ylabel(attr_name)
                ax.legend()
                ax.grid(True)

                self.crop_attr_figs[attr_name] = fig
                self.crop_attr_axes[attr_name] = ax
                self.crop_attr_lines[attr_name] = line
        plt.pause(1)

    def update_plot(self, t, y, x_lim, crop_attrs=None):
        # Convert the limit from seconds to days
        x_lim_days = x_lim / (24 * 3600)  # Convert x_lim from seconds to days

        for i, ax in enumerate(self.env_axes):
            days = t / (24 * 3600)  # Convert seconds to days
            self.lines[i].set_data(days, y[i, :])
            ax.set_xlim(0, x_lim_days)  # Adjust x_lim to days

            y_min, y_max = y[i, :].min(), y[i, :].max()
            ax.set_ylim(y_min-10, y_max+10)

        for i, ax in enumerate(self.crop_axes):
            j = i + len(self.env_axes)
            days = t / (24 * 3600)  # Convert seconds to days
            self.lines[j].set_data(days, y[j, :])
            ax.set_xlim(0, x_lim_days)  # Adjust x_lim to days

            y_min, y_max = y[j, :].min(), y[j, :].max()
            ax.set_ylim(y_min-10, y_max+10)

        if crop_attrs is not None:
            days = t / (24 * 3600)
            for attr_name, data in crop_attrs.items():
                self.crop_attr_lines[attr_name].set_data(days, data)
                ax = self.crop_attr_axes[attr_name]
                ax.set_xlim(0, x_lim_days)  # Adjust x_lim to days
                ax.relim()
                ax.autoscale_view()
        
        plt.pause(0.1)
        
