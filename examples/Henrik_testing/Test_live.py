import gym
import numpy as np
import pandas as pd

# Initialize the environment
env = gym.make('YourEnv-v0')
obs = env.reset()

# Initialize lists to store data
observations = []
actions = []
rewards = []
dones = []
infos = []

# Run the simulation loop
done = False
while not done:
    action = env.action_space.sample()  # Replace with your action selection logic
    obs, reward, done, info = env.step(action)

    # Store the data
    observations.append(obs)
    actions.append(action)
    rewards.append(reward)
    dones.append(done)
    infos.append(info)

# Convert lists to a DataFrame
data = {
    'observations': observations,
    'actions': actions,
    'rewards': rewards,
    'dones': dones,
    'infos': infos
}
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('simulation_results.csv', index=False)