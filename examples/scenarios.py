import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Set path for running local script
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Ignore tensorflow warning

import datetime
from config.utils import load_config
from data.utils import load_data
from envs.env import VerticalFarmEnv

import numpy as np

config = load_config('../config/')

# Winter scenario
# start_date = '2023-02-05'
# end_date = '2023-02-25'

# Summer scenario
start_date = '2023-07-10'
end_date = '2023-07-30'

start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')

data = load_data('../data', start_datetime, end_datetime)

# Generate dummy lighting PPFD control input
num_days = (end_datetime - start_datetime).days
base_pattern = np.concatenate([np.full(18, 1), np.full(6, 0)])
PPFDs = np.tile(base_pattern, num_days)
PPFDs = np.append(PPFDs, 0)

eval_env = VerticalFarmEnv(start_datetime, config, data, render_mode='file')

obs, _ = eval_env.reset()
i = 0
while True:
    u_sup = 1
    PPFD = PPFDs[i]
    action = np.array([0, u_sup, 0, 0, 0, 0, PPFD])

    obs, rewards, terminated, truncated, info = eval_env.step(action)
    eval_env.render()

    if terminated:
        eval_env.close()
        break

    i += 1
