from render.plot import plot_climate_figure, plot_crop_figure, plot_control_input_figure, plot_Qs, plot_Phis
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file into a DataFrame
csv_file_path = 'simulation_results.csv'
df = pd.read_csv(csv_file_path)

# Extract data from the DataFrame
dates = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
y = df[[col for col in df.columns if col.startswith('obs_')]].values.T
print(y)
climate_attrs = y[11:19, :]
crop_attrs = y[6:11, :]
actions = y[:7, :]
#print(actions)
all_data = y[19:23, :]
Qs = df[[col for col in df.columns if col.startswith('Q_')]].values.T
Phis = df[[col for col in df.columns if col.startswith('Phi_')]].values.T
#print(Phis)
print(Qs)

# Call the plotting functions with the extracted data
plot_climate_figure(dates, y, climate_attrs, all_data)
plot_crop_figure(dates, y, crop_attrs)
plot_control_input_figure(dates, actions)
plot_Qs(dates, Qs)
plot_Phis(dates, Phis)

# Display the plots
plt.show()