#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

import numpy as np
import matplotlib.pyplot as plt
from crop import CropModel
from environment import EnvironmentModel
from utilities import *
import json
from casadi import SX, vertcat, exp, sum1
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
from opt_crop_model import export_biomass_ode_model
from opt_setup_solver import opt_setup
from opt_utils import plot_crop, generate_energy_price, generate_photoperiod_values, print_ocp_setup_details
# import acados.interfaces.acados_template as at
def load_config(file_path: str) -> dict:
    """Load configuration from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)
def main():
    crop_config = load_config('crop_model_config.json')
    env_config = load_config('env_model_config.json')
    opt_config = load_config('opt_config.json')
    ocp_type = opt_config['ocp_type']
    N_horizon = opt_config['N_horizon']
    photoperiod = opt_config['photoperiod']
    darkperiod = opt_config['darkperiod']
    energy_prices = generate_energy_price(N_horizon=N_horizon)
    photoperiod_values = generate_photoperiod_values(photoperiod=photoperiod, darkperiod=darkperiod, N_horizon=N_horizon)
    Crop = CropModel(crop_config)
    Env = EnvironmentModel(env_config)
    use_RTI = False
    x0 = np.array([Crop.X_ns, Crop.X_s, 0, 0])
    Fmax = opt_config['Fmax']   # Fmax: The maximum allowed input
    Fmin = opt_config['Fmin']   # Fmin: The minimum allowed input
    DaysSim = opt_config['DaysSim']
    Tsim = DaysSim * 24 * 3600   # Tsim: The total time of the whole simulation (closed loop)
    Ts = opt_config['Ts']       # Ts: the sampling time of one step
    N_horizon = opt_config['N_horizon'] # N_horizon: The number of steps within the horizon
    Tf = N_horizon * Ts         # Tf: The total time of the horizon
    ocp_solver, integrator, ocp = opt_setup(Crop=Crop, Env=Env, opt_config=opt_config, energy_prices=energy_prices, photoperiod_values=photoperiod_values, x0=x0
                                       , Fmax=Fmax, Fmin=Fmin, N_horizon=N_horizon, Tf=Tf, ocp_type=ocp_type,RTI=use_RTI)
    
    
    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu
    Nsim = int(np.floor(Tsim/Ts))
    simX = np.zeros((Nsim+1, nx))
    simU = np.zeros((Nsim, nu))

    status = ocp_solver.solve()
    ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    if status != 0:
        raise Exception(f'acados returned status {status}.')

    # get solution
    for i in range(Nsim):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[Nsim,:] = ocp_solver.get(N_horizon, "x")

    plot_crop(np.linspace(0, Tf, Nsim+1), Fmax, simU, simX, latexify=False)


if __name__ == '__main__':
    main()