from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from crop import CropModel
from environment import EnvironmentModel
from utilities import *
import json

from acados_template import AcadosOcp, AcadosOcpSolver
from opt_crop_model import export_biomass_ode_model
# import acados.interfaces.acados_template as at
def load_config(file_path: str) -> dict:
    """Load configuration from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)
crop_config = load_config('crop_model_config.json')
Crop = CropModel(crop_config)

def main():
    # Create the ocp object
    ocp = AcadosOcp()

    # set model
    model = export_biomass_ode_model(Crop=Crop, uninh_air_vel=0.2)
    ocp.model = model

    Tf = 60
    nx = model.x.rows()
    nu = model.u.rows()
    N  = 10

    ocp.dims.N = N

    Q_mat = np.diag([0,0])
    R_mat = 1*np.diag([1])



"""
# Define CasADi symbols for states and control inputs
X = SX.sym('X', 2)  # State vector: [X_ns, X_s]
U = SX.sym('U', 1)  # Control input: [PPFD]

# Define the continuous dynamics model using CasADi
f_cont_dyn = Function('f_cont_dyn', [X, U], [biomass_ode_mpc_acados(X, U)])


# Create an acados OCP (Optimal Control Problem) object
ocp = at.acados_ocp_nlp()

# Set the model
ocp.model = f_cont_dyn

# Configure the OCP settings (horizon, constraints, objective, etc.)

# Example: Setting the horizon length
ocp.solver_options.tf = 10  # Prediction horizon

# Define constraints and objective function based on your requirements

# Compile the OCP solver
ocp_solver = at.acados_ocp_solver(ocp, json_file='acados_ocp.json')

# Solve the MPC problem with initial conditions and reference
x0 = [Crop.X_ns, Crop.X_s]  # Initial state
ocp_solver.set(0, 'lbx', x0)  # Set lower bound of states to initial condition
ocp_solver.set(0, 'ubx', x0)  # Set upper bound of states to initial condition

# Solve the OCP
status = ocp_solver.solve()

# Retrieve the optimal control input
u_opt = ocp_solver.get(0, 'u')


"""