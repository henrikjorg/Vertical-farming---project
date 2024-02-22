import matplotlib.pyplot as plt
from config import *

class Plotter:
    def __init__(self, t, y):
        plt.ion()

        # Environment figure
        self.env_fig, self.env_axes = plt.subplots(len(env_states_info), 1, figsize=(10, len(env_states_info)*3))
        plt.subplots_adjust(hspace=0.5)
        self.env_fig.suptitle('Environment')
        self.lines = []
        for i, (ax, (_, info)) in enumerate(zip(self.env_axes, env_states_info.items())):
            line = ax.plot(t,y[i,:])[0]
            ax.set_title(info.get('title', ''))
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(info.get('title', '-') + ' [' + info['unit'] + ']')
            ax.set_ylim(0,100)
            ax.set_xlim(0,1)
            ax.grid(True)
            
            self.lines.append(line)

        # Crop figure
        self.crop_fig, self.crop_axes = plt.subplots(len(crop_states_info), 1, figsize=(10, len(crop_states_info)*3))
        plt.subplots_adjust(hspace=0.5)
        self.crop_fig.suptitle('Crop')
        for i, (ax, (_, info)) in enumerate(zip(self.crop_axes, crop_states_info.items())):
            line = ax.plot(t,y[i,:],info.get('color', 'b'))[0]
            ax.set_title(info.get('title', ''))
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(info.get('title', '-') + ' [' + info['unit'] + ']')
            ax.set_ylim(0,100)
            ax.set_xlim(0,1)
            ax.grid(True)
            
            self.lines.append(line)

        plt.pause(1)

    def update_plot(self, t, y, x_lim):
        for i, ax in enumerate(self.env_axes):
            self.lines[i].set_data(t, y[i, :])
            ax.set_xlim(0,x_lim)

            y_min, y_max = y[i, 0:x_lim].min(), y[i, 0:x_lim].max()
            ax.set_ylim(y_min-10, y_max+10)

        for i, ax in enumerate(self.crop_axes):
            j = i + len(self.env_axes)
            self.lines[j].set_data(t, y[j, :])
            ax.set_xlim(0,x_lim)

            y_min, y_max = y[j, 0:x_lim].min(), y[j, 0:x_lim].max()
            ax.set_ylim(y_min-10, y_max+10)

        plt.pause(0.25)
