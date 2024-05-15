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

start_date = '2023-04-01'
end_date = '2023-04-22'
start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')

data = load_data('../data', start_datetime, end_datetime)

eval_env = VerticalFarmEnv(start_datetime, config, data, end_datetime=end_datetime, render_mode='live')

obs, _ = eval_env.reset()
while True:
    T_in, Chi_in, CO2_in, T_env, T_sup, Chi_sup, X_ns, X_s = obs[:8]
    LAI, CAC, f_phot = obs[8:11]
    T_hvac, Chi_hvac, Chi_out, CO2_out, T_crop, T_desired, Chi_desired, CO2_desired = obs[11:19]
    T_out, RH_out, electricity_price, U_par = obs[19:23]

    # HYPOTHETICAL SCENARIOS
    # action = [u_rot, u_fan, u_cool, u_heat, u_humid, u_c_inj] (Normalized values 0-1)

    # TEST P CONTROLLER
    # T_error = T_desired - T_in

    # Kp = 0.2
    # u_heat = Kp*T_error
    # u_heat = max(0, min(1, u_heat))
    # # print("u_heat PD: ", u_heat)

    # # SCENARIO 1:
    # u_c_inj = 0
    # if CO2_in < CO2_desired:
    #     u_c_inj = 1

    # u_humid = 0
    # if Chi_in < Chi_desired:
    #     u_humid = 1


    # COOLER: Moisture control!!!

    # u_heat = 0
    # if T_in < T_desired - 1:
    #     u_heat = 1

    # u_fan = 0.5
    # if Chi_in > Chi_desired or T_in > T_desired + 1:
    #     u_fan = 1

    # action = np.array([1, u_fan, 0, u_heat, u_humid, u_c_inj])

    # SCENARIO 2: Cooling if temperature is above 25 degrees
    # if T_in > T_desired + 1:
    #     action = np.array([0, 0.1, 1, 0, 0, 0])
    # else:
    #     action = np.array([0, 0.1, 0, 0, 0, 0])

    # SCENARIO 3: Balanced CO2 injection
    # if CO2_in < CO2_desired - 100:
    #     action = np.array([0, 0.1, 0, 0, 0, 1])
    # else:
    #     action = np.array([0, 0.1, 0, 0, 0, 0])

    action = np.array([1, 1, 0, 0, 0, 0])

    obs, rewards, terminated, truncated, info = eval_env.step(action)
    eval_env.render()

    if terminated:
        eval_env.close()
        break
