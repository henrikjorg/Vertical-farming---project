import numpy as np
import matplotlib.pyplot as plt
from crop import CropModel
from environment import EnvironmentModel
from utilities import *
import json
from casadi import SX, vertcat, exp, sum1, dot
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from mpc_optimization.opt_crop_model import export_biomass_ode_model
INF = 1e12
def opt_setup(Crop, Env, opt_config, energy_prices, photoperiod_values, x0, Fmax, Fmin, N_horizon, Days_horizon, Ts,Tf, ocp_type="minimize_cost", RTI=False):


    # Create the ocp object
    ocp = AcadosOcp()
    light_percent = opt_config['photoperiod'] / (opt_config['photoperiod'] + opt_config['darkperiod'])
    # Set model
    model = export_biomass_ode_model(Crop=Crop, Env=Env, Ts=Ts, N_horizon=N_horizon, Days_horizon=Days_horizon, light_percent=light_percent)
    ocp.parameter_values = np.array([1,1])
    #model = export_pendulum_ode_model()

    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = N_horizon

    x_ref = np.array([0,0,0,0,0,0,0])#np.array(opt_config['x_ref'])
    Q_mat = np.diag([0, # NS mass                   0
                     0, # S mass                    1
                     0, # Fresh mass one shoot      2
                     0, # DLI                       3
                     1, # Average hourly cost       4
                     0, # Constraint on dark period 5
                     0])# Avg PPFD during photoperiod  6
    #Q_mat = np.diag(opt_config['Q_mat'])
    q_arr = np.array([0,0,0,0,0,0,0])#np.array(opt_config['q_arr'])
    R_mat = np.diag(opt_config['R_mat'])
    r_arr = np.array(opt_config['r_arr'])
    total_cost = model.u / (N_horizon * Ts)
    Cost_mat = np.diag([1e3])
    
    x_diff = model.x - x_ref
    if ocp_type == "lin_cost":
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.model.cost_expr_ext_cost =  0#total_cost.T @ Cost_mat @ total_cost# ca.dot(q_arr, model.x) #total_cost   #r_arr @ model.u + ca.dot(q_arr, model.x)
        ocp.model.cost_expr_ext_cost_e = x_diff.T @ Q_mat @ x_diff#ca.dot(q_arr,model.x) #x_diff.T @ Q_mat @ x_diff #+ ca.dot(q_arr,model.x)
        # set constraints 
        ocp.constraints.lbu = np.array([Fmin])
        ocp.constraints.ubu = np.array([Fmax])
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.x0 = x0
        # Constraints on the intermediate stages
        #       No light during dark period: 0,0 state_ind 5
        #       DLI constraint: -100, 
        #ocp.constraints.lbx = np.array([0])
        #ocp.constraints.ubx = np.array([0])
        #ocp.constraints.idxbx = np.array([5])
        # Constraints for final fresh mass
        #       DLI:                 100,500 state_ind 6
        #       Fresh biomass                k,l state_ind 2
        ocp.constraints.lbx_e = np.array([10])
        ocp.constraints.ubx_e = np.array([2000])
        ocp.constraints.idxbx_e = np.array([2])

        
    
    elif ocp_type == "lin_cost_lin_mass":
        # the 'EXTERNAL' cost type can be used to define general cost terms
        # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.model.cost_expr_ext_cost = model.x.T @ Q_mat @ model.x + model.u.T @ R_mat @ model.u + ca.dot(q_arr, model.x) + r_arr @ model.u 
        ocp.model.cost_expr_ext_cost_e = model.x.T @ Q_mat @ model.x
        # set constraints
        ocp.constraints.lbu = np.array([Fmin])
        ocp.constraints.ubu = np.array([+Fmax])
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.x0 = x0
    elif ocp_type == "max_mass_per_cost":
        # the 'EXTERNAL' cost type can be used to define general cost terms
        # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.model.cost_expr_ext_cost = ca.dot(q_arr, model.x) / (r_arr @ model.u + 1)
        ocp.model.cost_expr_ext_cost_e = 0#ca.dot(q_arr, model.x)
        # set constraints
        ocp.constraints.lbu = np.array([Fmin])
        ocp.constraints.ubu = np.array([+Fmax])
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.x0 = x0

    ocp.solver_options.qp_solver =  opt_config['qp_solver']
    ocp.solver_options.hessian_approx = opt_config['hessian_approx']
    ocp.solver_options.integrator_type = opt_config['integrator_type']
    ocp.solver_options.print_level = opt_config['print_level']
    ocp.solver_options.nlp_solver_type = opt_config['nlp_solver_type']

    ocp.solver_options.nlp_solver_max_iter = opt_config['nlp_solver_max_iter']

    ocp.solver_options.qp_solver_iter_max = opt_config["qp_solver_iter_max"]
    ocp.solver_options.nlp_solver_tol_stat = opt_config['nlp_solver_tol_stat']


    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'mpc_optimization/acados_ocp.json')

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp.json')

    return acados_ocp_solver, acados_integrator, ocp
