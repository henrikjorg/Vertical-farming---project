import numpy as np
import pandas as pd
import gymnasium as gym
from datetime import datetime

from gymnasium import spaces
from scipy.integrate import solve_ivp
from models.deterministic.model import Model
from typing import Optional
from render.live import RenderLive
from render.print import RenderPrint
from render.file import RenderFile
from config.utils import get_attribute

HOURS_IN_DAY = int(24)
MINUTES_IN_HOUR = int(60)
SECONDS_IN_MINUTE = int(60)

class VerticalFarmEnv(gym.Env):
    """A vertical farm environment that follows the OpenAI Gym interface"""
    metadata = {
        'render_modes': ['print', 'live', 'file'],
    }

    def __init__(self, start_datetime, config, data, end_datetime: Optional[datetime] = None, render_mode: Optional[str] = None):
        self.env_start_datetime = start_datetime
        self.env_end_datetime = end_datetime
        self.config = config
        self.data = data
        self.render_mode = render_mode

        # Get simulation configuration attributes
        self.cycle_duration_days = get_attribute(config, 'cycle_duration_days')

        self.u_sup_max = get_attribute(config, 'u_sup_max')
        self.rho_air = get_attribute(config, 'rho_air')
        self.c_air = get_attribute(config, 'c_air')

        # Initialize model
        self.model = Model(self.config, self.cycle_duration_days)

        # Set attributes to render
        self.crop_attributes_to_render = ['LAI', 'CAC', 'f_phot', 'dry_weight_per_plant', 'fresh_weight_shoot_per_plant']
        self.climate_attributes_to_render = ['T_hvac', 'Chi_hvac', 'Chi_out', 'CO2_out', 'T_crop', 'T_desired', 'Chi_desired', 'CO2_desired']

        # Initialize Gym environment observation space
        num_states = 6
        num_data = 3
        num_attrs = len(self.crop_attributes_to_render) + len(self.climate_attributes_to_render)
        num_obs = num_states + num_attrs + num_data

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float64)
        
        # Initialize normalized Gym environment action space
        num_control_inputs = 7
        self.action_space = spaces.Box(low=0, high=1, shape=(num_control_inputs,), dtype=np.float32)

        self._initialize_simulation(self.cycle_duration_days)
        
    def _initialize_simulation(self, num_days: int):
        self.terminated = False

        self.seconds_per_iter = MINUTES_IN_HOUR*SECONDS_IN_MINUTE
        self.Tf = num_days*HOURS_IN_DAY*self.seconds_per_iter               # Final time of the simulation
        self.t_eval = np.linspace(0,self.Tf, num=self.Tf+1, endpoint=True)  # Time points to evaluate the solution
        
        self.cur_index_i = 0 # Initial index of the current iteration

        # Initialize the state vector and an array to store the solutions
        self.y0 = np.concatenate((self.model.climate_model.init_state, self.model.crop_model.init_state), axis=None)
        self.solutions = np.zeros([len(self.y0), len(self.t_eval)], dtype=float)

        # Initialize arrays to store the control inputs and data
        self.actions = np.zeros([self.action_space._shape[0], len(self.t_eval)], dtype=float)
        self.all_data = np.zeros([len(self.data.columns), len(self.t_eval)], dtype=float)

        # Initialize dictionaries storing other attributes to render
        self.crop_attrs = {}
        self.climate_attrs = {}
        for attr in self.crop_attributes_to_render:
            self.crop_attrs[attr] = np.zeros(len(self.t_eval))
        for attr in self.climate_attributes_to_render:
            self.climate_attrs[attr] = np.zeros(len(self.t_eval))

    def _get_obs(self):
        state = self.y0

        crop_attributes = np.array([self.crop_attrs[attr][self.cur_index_i] for attr in self.crop_attributes_to_render])
        climate_attributes = np.array([self.climate_attrs[attr][self.cur_index_i] for attr in self.climate_attributes_to_render])

        data = self.data.iloc[self.end_index, :].values

        return np.concatenate((state, crop_attributes, climate_attributes, data))

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.visualization = None
        self.current_step = 0
        self.reward = 0.0
        self._initialize_simulation(self.cycle_duration_days)
        
        if self.env_end_datetime is None: # The simulation will only run for one cycle
            self.start_datetime = self.env_start_datetime

        else: # The simulation will run for multiple cycles, and randomly select a start date for each
            days = (self.env_end_datetime - self.env_start_datetime).days

            if days < self.cycle_duration_days:
                raise ValueError("The end date must be at least cycle duration days after the start date")
            if days == self.cycle_duration_days:
                self.start_datetime = self.env_start_datetime
            else: 
                # Random start date must be at least cycle duration days before the end date
                random_days = np.random.randint(0, days - self.cycle_duration_days) 
                self.start_datetime = self.env_start_datetime + pd.DateOffset(days=random_days)
        
        self.start_index = self.data.index.get_loc(self.start_datetime)
        self.current_index = self.data.index.get_loc(self.start_datetime)
        self.end_index = self.start_index + self.cycle_duration_days*HOURS_IN_DAY
        
        self.observation = self._get_obs()
        
        return self.observation, {}

    def step(self, action):
        action = self.model.denormalize_action(action)
        control_input = tuple(action)

        data = self.data.iloc[self.current_index, :].values
        external_input = tuple(data)
        
        # Calculate HVAC supply air based on current state and action
        T_hvac, Chi_hvac, Chi_out = self.model.climate_model.hvac_model.calculate_supply_air(self.y0, control_input, data)
        hvac_input = tuple([T_hvac, Chi_hvac, Chi_out])
        
        sol = solve_ivp(fun=self.model.ODEs,
                        t_span=[0, self.seconds_per_iter],
                        y0=self.y0,
                        method='RK45',
                        t_eval=np.linspace(0, self.seconds_per_iter, num=self.seconds_per_iter),
                        args=[control_input, external_input, hvac_input, self.current_step])
        
        self.process_solution(sol, action, data)

        self.cur_index_i += len(sol.t) # Update current index for next iteration
        self.y0 = sol.y[:,-1] # Update initial conditions for next iteration

        self.current_index += 1
        self.current_step += 1

        self.terminated = bool(self.cur_index_i == self.Tf)
        if self.terminated:
            self.solutions[:,-1] = sol.y[:,-1] # The last solution should not be zero (for rendering purposes)

        self.observation = self._get_obs()

        self.reward = 0.0

        return self.observation, self.reward, self.terminated, False, {}
    
    def process_solution(self, sol, action, data):
        self.cur_index_f = self.cur_index_i + len(sol.t)

        # Store the solution in the solutions array
        self.solutions[:,self.cur_index_i:self.cur_index_f] = sol.y

        # Store the control inputs and data
        self.actions[:,self.cur_index_i:self.cur_index_f+1] = np.tile(action, (self.seconds_per_iter+1, 1)).T
        self.all_data[:,self.cur_index_i:self.cur_index_f+1] = np.tile(data, (self.seconds_per_iter+1, 1)).T
        
        # Store the attributes to render
        for key in self.crop_attrs.keys():
            self.crop_attrs[key][self.cur_index_i:self.cur_index_f+1] = getattr(self.model.crop_model, key)
        for key in self.climate_attrs.keys():
            self.climate_attrs[key][self.cur_index_i:self.cur_index_f+1] = getattr(self.model.climate_model, key)
    
    def render(self):
        if self.render_mode == None:
            return
        
        elif self.render_mode == 'print':
            if self.visualization == None:
                self.visualization = RenderPrint(self.start_datetime)

        elif self.render_mode == 'live':
            if self.visualization == None:
                self.visualization = RenderLive(self.config, self.start_datetime, self.t_eval, self.solutions)

        elif self.render_mode == 'file':
            if self.visualization == None:
                self.visualization = RenderFile(self.config, self.start_datetime, self.t_eval, self.solutions)

        self.visualization.render(self.terminated, self.current_step, self.t_eval, self.model, self.solutions, self.climate_attrs, self.crop_attrs, self.actions, self.all_data, self.cur_index_i)
        
    def close(self):
        # NB! This is a temporary solution to fill the zero values of Q and Phi for plotting
        # The arrays storing the intermediate values of Q and Phi from inside the solve_ivp function are not fully filled

        self.model.climate_model.Q_data[0,:] = fill_zeros_with_last(self.model.climate_model.Q_data[0,:])
        self.model.climate_model.Q_data[1,:] = fill_zeros_with_last(self.model.climate_model.Q_data[1,:])
        self.model.climate_model.Q_data[2,:] = fill_zeros_with_last(self.model.climate_model.Q_data[2,:])
        self.model.climate_model.Q_data[3,:] = fill_zeros_with_last(self.model.climate_model.Q_data[3,:]) # Q_hvac

        # Set values of Q_light where PPFD = 0 to 0
        for i in range(len(self.t_eval)):
            if self.actions[6,i] == 0:
                self.model.climate_model.Q_data[2,i] = 0

        self.model.climate_model.Phi_data[0,:] = fill_zeros_with_last(self.model.climate_model.Phi_data[0,:])
        self.model.climate_model.Phi_data[1,:] = fill_zeros_with_last(self.model.climate_model.Phi_data[1,:])
        self.model.climate_model.Phi_c_data[0,:] = fill_zeros_with_last(self.model.climate_model.Phi_c_data[0,:])
        self.model.climate_model.Phi_c_data[1,:] = fill_zeros_with_last(self.model.climate_model.Phi_c_data[1,:])
        self.model.climate_model.Phi_c_data[2,:] = fill_zeros_with_last(self.model.climate_model.Phi_c_data[2,:])

        # Calculate balanced values of T_sup and Chi_sup
        for i in range(len(self.t_eval)):
            self.solutions[4,i] = self.solutions[0,i] + self.model.climate_model.Q_data[3,i]/(self.u_sup_max*self.rho_air*self.c_air)
            self.solutions[5,i] = self.solutions[1,i] + self.model.climate_model.Phi_data[1,i]/(self.u_sup_max)

        if self.render_mode == 'file':
            self.visualization.save(self.t_eval, self.model, self.solutions, self.climate_attrs, self.crop_attrs, self.actions, self.all_data)
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None

def fill_zeros_with_last(arr):
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]
