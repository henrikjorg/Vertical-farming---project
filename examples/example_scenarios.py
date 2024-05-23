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
base_pattern = np.concatenate([np.full(16, 1), np.full(8, 0)])
PPFDs = np.tile(base_pattern, num_days)
PPFDs = np.append(PPFDs, 0)

# eval_env = VerticalFarmEnv(start_datetime, config, data, end_datetime=end_datetime, render_mode='file')
eval_env = VerticalFarmEnv(start_datetime, config, data, render_mode='file')

obs, _ = eval_env.reset()
i = 0
while True:
    T_in, Chi_in, CO2_in, T_env, T_sup, Chi_sup, X_ns, X_s = obs[:8]
    LAI, CAC, f_phot = obs[8:11]
    T_hvac, Chi_hvac, Chi_out, CO2_out, T_crop, T_desired, Chi_desired, CO2_desired = obs[11:19]
    T_out, RH_out, electricity_price = obs[19:22]

    #############################################
    #                                           #
    #                 PD control                #
    #                                           #
    #############################################

    # TEST P CONTROLLER
    T_error = T_desired - T_in
    Chi_error = Chi_desired - Chi_in
    CO2_error = CO2_desired - CO2_in

    u_humid = 0
    # Kp_humid = 300
    # u_humid = Kp_humid*Chi_error
    # u_humid = max(0, min(1, u_humid))

    u_sup = 0
    # combined_error = abs(T_error) + abs(Chi_error)
    # Kp_sup = 0.1
    # u_sup = Kp_sup*combined_error
    # u_sup = max(0, min(1, u_sup))
    u_sup = 1

    u_rot = 0.5
    if T_error > 0 or Chi_error > 0:
        u_rot = 1

    u_c_inj = 0
    # Kp_c_inj = 0.1
    # u_c_inj = Kp_c_inj*CO2_error
    # u_c_inj = max(0, min(1, u_c_inj))

    u_heat = 0
    # Kp_heat = 0.2
    # u_heat = Kp_heat*T_error
    # u_heat = max(0, min(1, u_heat))

    u_cool = 0
    # Kp_cool = 0.2
    # u_cool = Kp_cool*T_error
    # u_cool = min(0, max(-1, u_cool))
    # u_cool = -u_cool


    # PRINT
    # print("u_rot P: ", u_rot)
    # print("u_sup P: ", u_sup)
    # print("u_cool P: ", u_cool)
    # print("u_heat P: ", u_heat)
    # print("u_humid P: ", u_humid)
    # print("u_c_inj P: ", u_c_inj)
    # print()


    PPFD = PPFDs[i]

    action = np.array([u_rot, u_sup, u_cool, u_heat, u_humid, u_c_inj, PPFD])

    obs, rewards, terminated, truncated, info = eval_env.step(action)
    eval_env.render()

    if terminated:
        eval_env.close()
        break

    i += 1
