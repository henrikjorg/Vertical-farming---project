#######################################################################
# DEPRECATED SCRIPT WARNING
# This script is deprecated.
#
# Reason for deprecation:
# - The current implementation of the Gymnasium environment does not implement a suitable reward function for reinforcement learning.
#
# Reference Documentation:
# - Reinforcement Learning Tips and Tricks: https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html
# - Stable Baselines3 PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
# - Examples: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
#######################################################################

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Set path for running local script
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Ignore tensorflow warning

import datetime
from config.utils import load_config, get_attribute
from data.utils import load_data
from envs.env import VerticalFarmEnv

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

'''This script trains a PPO model using the VerticalFarmEnv environment and evaluates it on a new dataset'''

config = load_config('../config/')
cycle_duration_days = get_attribute(config, 'cycle_duration_days')

# Train model
train_start_date = '2020-01-01'
train_end_date = '2020-03-03'
train_start_datetime = datetime.datetime.strptime(train_start_date, '%Y-%m-%d')
train_end_datetime = datetime.datetime.strptime(train_end_date, '%Y-%m-%d')

train_data = load_data('../data', train_start_datetime, train_end_datetime)

n_envs = 4
training_env = make_vec_env(lambda: VerticalFarmEnv(train_start_datetime, config, train_data, end_datetime=train_end_datetime), n_envs=n_envs)

check_env(training_env)

n_steps = cycle_duration_days*24
total_episodes = 10
total_timesteps = n_steps*n_envs*total_episodes

# Train model and log to tensorboard
model = PPO("MlpPolicy", training_env, n_steps=n_steps, batch_size=n_steps*n_envs, verbose=1, tensorboard_log="./ppo_vertical_farm_tensorboard/")
model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name="test", log_interval=1)

model.save("vertical_farm_test_long")
del model # delete trained model to demonstrate loading

# Evaluate model
model = PPO.load("vertical_farm_test_long")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

eval_start_date = '2022-06-06'
eval_start_datetime = datetime.datetime.strptime(eval_start_date, '%Y-%m-%d')
eval_end_datetime = eval_start_datetime + datetime.timedelta(days=cycle_duration_days)

eval_data = load_data('../data', eval_start_datetime, eval_end_datetime)

n_envs = 1
eval_env = make_vec_env(lambda: VerticalFarmEnv(eval_start_datetime, config, eval_data, render_mode='file'), n_envs=n_envs)

model.set_env(eval_env)

obs = eval_env.reset()
while True:
    action, _states = model.predict(obs)

    obs, rewards, terminated, info = eval_env.step(action)
    eval_env.render()

    if terminated:
        eval_env.close()
        break