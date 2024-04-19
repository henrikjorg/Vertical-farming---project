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
def opt_setup(Crop, Env, opt_config, energy_prices, photoperiod_values, x0, Fmax, Fmin, N_horizon, Ts,Tf, ocp_type="minimize_cost", RTI=False):


    # Create the ocp object
    ocp = AcadosOcp()
    photoperiod_length = opt_config['photoperiod_length'] 
    darkperiod_length = opt_config['darkperiod_length']
    min_DLI = opt_config['min_DLI']
    max_DLI = opt_config['max_DLI']
    # Set model
    model = export_biomass_ode_model(Crop=Crop, Env=Env, Ts=Ts, N_horizon=N_horizon,
                                      photoperiod_length=photoperiod_length, darkperiod_length=darkperiod_length, min_DLI=min_DLI)
    ocp.parameter_values = np.array([1,1, 1])
    #model = export_pendulum_ode_model()

    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = N_horizon

    x_ref = np.array([0,    # NS mass
                      0,    #S mass
                      1000,    # Fresh mass shoot
                      0,    # DLI (one pp per day!)
                      0,    # DLI lower
                      0,    # Average hourly cost
                      0])   # Avg PPFD during pp
    #x_ref = np.array(opt_config['x_ref'])
    Q_mat = np.diag([0, # NS mass                   0
                     0, # S mass                    1
                     1, # Fresh mass one shoot      2
                     0, # DLI uppert (one pp per day!)     3
                     0, # DLI lower bound           4
                     0, # Average hourly cost       5
                     0])# Avg PPFD during photoperiod  6
    
    #Q_mat = np.diag(opt_config['Q_mat'])
    q_arr = np.array([0,0,0,0,0,0,0])#np.array(opt_config['q_arr'])
    R_mat = np.diag(opt_config['R_mat'])
    r_arr = np.array(opt_config['r_arr'])
    
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
        #       DLI constraint: -100, 
        #ocp.constraints.lbx = np.array([-1, -1])
        #ocp.constraints.ubx = np.array([max_DLI, 10000])
        #ocp.constraints.idxbx = np.array([3, 4])
        #
        #ocp.constraints.lbx = np.array([-1, -1])
        #ocp.constraints.ubx = np.array([max_DLI,10000])
        #ocp.constraints.idxbx = np.array([3,4])

        
        #
        #
        tol = 100000
        ocp.constraints.lh = np.array([0, 0, 0])
        ocp.constraints.uh = np.array([0, max_DLI, max_DLI])
        ocp.constraints.lh_0 = np.array([0, 0, 0])
        ocp.constraints.uh_0 = np.array([0, max_DLI, max_DLI])
        #ocp.constraints.lh_e = np.array([0])
        #ocp.constraints.uh_e = np.array([0])
        # Constraints for final fresh mass
        #       DLI:                 100,500 state_ind 5
        #       Fresh biomass                k,l state_ind 2
        #ocp.constraints.lbx_e = np.array([70])
        #ocp.constraints.ubx_e = np.array([max_DLI, 10000])
        #ocp.constraints.idxbx_e = np.array([3, 4])

        
    
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
    #ocp.solver_options.nlp_solver_type = opt_config['nlp_solver_type']

    #ocp.solver_options.nlp_solver_max_iter = opt_config['nlp_solver_max_iter']

    #ocp.solver_options.qp_solver_iter_max = opt_config["qp_solver_iter_max"]
    #ocp.solver_options.nlp_solver_tol_stat = opt_config['nlp_solver_tol_stat']


    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'mpc_optimization/acados_ocp.json')

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = 'mpc_optimization/hello_acados_integrator.json')

    return acados_ocp_solver, acados_integrator, ocp
