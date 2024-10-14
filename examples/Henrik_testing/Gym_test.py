import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from config.utils import load_config
from external.utils import load_data
from envs.env import VerticalFarmEnv

# Set path for running local script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Ignore tensorflow warning

# Print the current working directory for debugging
print("Current working directory:", os.getcwd())

# Load configuration
config = load_config('../../config/')

# Define start and end datetime for the simulation
start_date = '2023-01-01'
end_date = '2023-01-31'
start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

# Correct the path to the data file
data_path = os.path.abspath('../../external/weather/data')
print("Data path:", data_path)

# Load data
data = load_data(data_path, start_datetime, end_datetime)

# Initialize the environment
env = VerticalFarmEnv(start_datetime=start_datetime, config=config, data=data, end_datetime=end_datetime, render_mode='print')

# Reset the environment to get the initial observation
observation, _ = env.reset()

# Lists to store simulation data
observations = []
rewards = []
Q_data = []
Phi_data = []

# Run the simulation for a number of steps
for step in range(100):  # Adjust the number of steps as needed
    action = env.action_space.sample()  # Sample a random action
    observation, reward, terminated, truncated, info = env.step(action)
    # Store the observation, reward, Q, and Phi
    observations.append(observation)
    rewards.append(reward)
    Q_data.append(env.model.climate_model.Q_data[:, env.cur_index_i])
    Phi_data.append(env.model.climate_model.Phi_data[:, env.cur_index_i])

    if terminated:
        break

# Close the environment
env.close()

# Convert lists to numpy arrays for saving
observations = np.array(observations)
rewards = np.array(rewards)
Q_data = np.array(Q_data)
Phi_data = np.array(Phi_data)

# Save the parameters and results to a CSV file
def save_simulation_results_to_csv(env, observations, rewards, Q_data, Phi_data, filename='simulation_results.csv'):

    # Create DataFrame for observations, rewards, Q_data, and Phi_data
    df_results = pd.DataFrame(observations, columns=[f'obs_{i}' for i in range(observations.shape[1])])
    df_results['reward'] = rewards
    df_results = pd.concat([df_results, pd.DataFrame(Q_data, columns=[f'Q_{i}' for i in range(Q_data.shape[1])])], axis=1)
    df_results = pd.concat([df_results, pd.DataFrame(Phi_data, columns=[f'Phi_{i}' for i in range(Phi_data.shape[1])])], axis=1)

    # Concatenate parameters and results
    df = pd.concat([df_results], axis=1)

    # Write to CSV
    df.to_csv(filename, index=False)

# Save the results
save_simulation_results_to_csv(env, observations, rewards, Q_data, Phi_data, 'simulation_results.csv')