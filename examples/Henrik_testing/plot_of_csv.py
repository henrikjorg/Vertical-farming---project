import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
csv_file_path = 'simulation_results.csv'
df = pd.read_csv(csv_file_path)


# Define observation categories and their labels
state_vector = [f'obs_{i}' for i in range(6)]
state_vector_labels = ['T_in', 'Chi_in', 'CO2_in', 'T_env', 'T_sup', 'Chi_sup']

crop_attributes = [f'obs_{i}' for i in range(6, 11)]
crop_attributes_labels = ['LAI', 'CAC', 'f_phot', 'dry_weight_per_plant', 'fresh_weight_shoot_per_plant']

climate_attributes = [f'obs_{i}' for i in range(11, 19)]
climate_attributes_labels = ['T_hvac', 'Chi_hvac', 'Chi_out', 'CO2_out', 'T_crop', 'T_desired', 'Chi_desired', 'CO2_desired']

external_data = [f'obs_{i}' for i in range(19, 23)]
external_data_labels = ['T_out', 'RH_out', 'Electricity price', 'Light input']

Q_data = [f'Q_{i}' for i in range(0, 5)]
Q_data_labels = ['Q_env', 'Q_sens_plant', 'Q_light', 'Q_hvac', 'P_light']

# Plot state vector
df_state_vector = df[state_vector]

plt.figure(figsize=(10, 6))
for column, label in zip(df_state_vector.columns, state_vector_labels):
    plt.plot(df.index, df_state_vector[column], label=label)
plt.xlabel('Step')
plt.ylabel('State Vector Value')
plt.title('State Vector')
plt.legend()
plt.show()

# Plot crop attributes
df_crop_attributes = df[crop_attributes]

plt.figure(figsize=(10, 6))
for column, label in zip(df_crop_attributes.columns, crop_attributes_labels):
    plt.plot(df.index, df_crop_attributes[column], label=label)
plt.xlabel('Step')
plt.ylabel('Crop Attribute Value')
plt.title('Crop Attributes')
plt.legend()
plt.show()

# Plot climate attributes
df_climate_attributes = df[climate_attributes]

plt.figure(figsize=(10, 6))
for column, label in zip(df_climate_attributes.columns, climate_attributes_labels):
    plt.plot(df.index, df_climate_attributes[column], label=label)
plt.xlabel('Step')
plt.ylabel('Climate Attribute Value')
plt.title('Climate Attributes')
plt.legend()
plt.show()

# Plot external data
df_external_data = df[external_data]

plt.figure(figsize=(10, 6))
for column, label in zip(df_external_data.columns, external_data_labels):
    plt.plot(df.index, df_external_data[column], label=label)
plt.xlabel('Step')
plt.ylabel('External Data Value')
plt.title('External Data')
plt.legend()
plt.show()

# Plot rewards
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['reward'], label='Reward', linestyle='--')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Rewards')
plt.legend()
plt.show()

# Plot Q data
df_Q_data = df[Q_data]

plt.figure(figsize=(10, 6))
for column, label in zip(df[Q_data].columns, Q_data_labels):
    plt.plot(df.index, df[column], label=label)
plt.xlabel('Step')
plt.ylabel('Q Value')
plt.title('Q Data')
plt.legend()
plt.show()

# Plot Phi data
Phi_columns = [col for col in df.columns if col.startswith('Phi_')]
df_Phi_data = df[Phi_columns]

plt.figure(figsize=(10, 6))
for column in df_Phi_data.columns:
    plt.plot(df.index, df_Phi_data[column], label=column)
plt.xlabel('Step')
plt.ylabel('Phi Value')
plt.title('Phi Data')
plt.legend()
plt.show()