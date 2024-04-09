import numpy as np
import matplotlib.pyplot as plt
from crop import CropModel
from environment import EnvironmentModel
from utilities import *
import json
from casadi import SX, vertcat, exp, sum1
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from opt_crop_model import export_biomass_ode_model
from pendulum_model import export_pendulum_ode_model
import scipy.linalg
from energy_prices import energy_price_array
def opt_setup(Crop, Env, opt_config, energy_prices, x0, Fmax, Fmin, N_horizon, Tf, RTI=False):


    # Create the ocp object
    ocp = AcadosOcp()

    # Set model
    model = export_biomass_ode_model(Crop=Crop, Env=Env)
    #model = export_pendulum_ode_model()

    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = N_horizon

    # Setting up the energy price
    #energy_price = SX.sym('energy_price', N_horizon)
    #ocp.model.p = energy_price
    #ocp.parameter_values = energy_prices
    #total_cost_expr = sum1(u * energy_price)
    #ocp.cost.cost_type = 'EXTERNAL'
    #ocp.cost.cost_expr_ext_cost = total_cost_expr
    
    # set cost module
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    Q_mat = 2*np.diag([1e3, 1e3, 1e2])*0
    R_mat = 2*np.diag([1e-2])

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat*0

    ocp.model.cost_y_expr = vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref  = np.array([0,0,0,500])
    ocp.cost.yref_e = np.ones((ny_e, ))

    # set constraints
    ocp.constraints.lbu = np.array([Fmin])
    ocp.constraints.ubu = np.array([Fmax])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.x0 = x0
    

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_newton_iter = 10

    if RTI:
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    else:
        ocp.solver_options.nlp_solver_type = 'SQP'

    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf

    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return acados_ocp_solver, acados_integrator
def opt_setup_pendulum(Crop, Env, opt_config, energy_prices, x0, Fmax, N_horizon, Tf, RTI=False):


    # Create the ocp object
    ocp = AcadosOcp()

    # Set model
    model = export_biomass_ode_model(Crop=Crop, Env=Env)
    model = export_pendulum_ode_model()

    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = N_horizon

    # set cost module
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    Q_mat = 2*np.diag([1e3, 1e3, 1e-2, 1e-2])
    R_mat = 2*np.diag([1e-2])

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat

    ocp.model.cost_y_expr = vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref  = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e, ))

    # set constraints
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])

    ocp.constraints.x0 = x0
    ocp.constraints.idxbu = np.array([0])

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_newton_iter = 10

    if RTI:
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    else:
        ocp.solver_options.nlp_solver_type = 'SQP'

    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf

    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return acados_ocp_solver, acados_integrator


def opt_setup2(Crop, Env, opt_config, energy_prices):
    ocp.model = model
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    N = opt_config['N_horizon']
    ocp.dims.N = N
    

    # Define the cost function
    u = ocp.model.u  # Control input
    """energy_price = SX.sym('energy_price', N)
    ocp.model.p = energy_price
    ocp.dims.np = N
    total_cost_expr = sum1(u * energy_price)
    custom_cost_expr = total_cost_expr
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_expr_ext_cost = custom_cost_expr
    ocp.parameter_values = energy_prices
    """
    # Set horizon
    ocp.dims.N = N
    ocp.solver_options.tf = N
    # Define constraints
    # Light energy cumulative sum should not exceed k_1
    k_1 = 1000  # Example value, adjust as necessary
    ocp.constraints.lbx = np.array([-ca.inf, -ca.inf, 0])  # Lower bounds for X_ns, X_s, z_cumsum
    ocp.constraints.ubx = np.array([ca.inf, ca.inf, k_1])  # Upper bounds, setting maximum for z_cumsum
    ocp.constraints.idxbx = np.array([0,1,2])
    # Final biomass constraint (X_ns_end + X_s_end must be higher than k_2)
    k_2 = 200  # Example value, adjust as necessary
    ocp.constraints.x0 = np.array([10, 10, 0])  # Initial state: adjust based on your system's initial condition
    
    # Control constraints (photoperiod)
    ocp.constraints.lbu = np.zeros((N,))  # Lower bound on control (off)
    ocp.constraints.ubu = np.ones((N,)) * 1000  # Upper bound on control (max PPFD)
    ocp.constraints.idxbu = np.array([0]*N)
    # Solve the OCP
    
    # Testing implementation
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    Q_mat = 2*np.diag([1e3, 1e3, 1e-2])
    R_mat = 2*np.diag([1e-2])

    ocp.model.cost_y_expr = vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref  = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e, ))

    ocp.constraints.x0 = np.array([10, 10, 0])
    # Control constraints (photoperiod)
    ocp.constraints.lbu = np.array([0])  # Lower bound on control (off)
    ocp.constraints.ubu = np.array([1000])  # Upper bound on control (max PPFD)
    ocp.constraints.idxbu = np.array([0])

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_newton_iter = 10


    ocp.solver_options.nlp_solver_type = 'SQP'

    ocp.solver_options.qp_solver_cond_N = N
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
        

    for i in range(N):
        # Set the parameter for the current timestep (energy price)
        acados_ocp_solver.set(i, 'p', np.array([energy_prices[i]]))  # energy_prices needs to be defined based on your data
    print(ocp.parameter_values)
    acados_integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp.json')

    return acados_ocp_solver, acados_integrator