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
def opt_setup(Crop, Env, opt_config, x0, Fmax, Fmin, N_horizon, Ts,Tf, ocp_type="minimize_cost", RTI=False):


    # Create the ocp object
    ocp = AcadosOcp()
    photoperiod_length = opt_config['photoperiod_length'] 
    darkperiod_length = opt_config['darkperiod_length']
    min_DLI = opt_config['min_DLI']
    max_DLI = opt_config['max_DLI']
    # Set model
    model = export_biomass_ode_model(Crop=Crop, Env=Env, Ts=Ts, N_horizon=N_horizon,
                                      photoperiod_length=photoperiod_length, darkperiod_length=darkperiod_length, min_DLI=min_DLI)
    ocp.parameter_values = np.array([1,1, 1, 1])
    #model = export_pendulum_ode_model()

    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()


    ocp.dims.N = N_horizon

    x_ref = np.array([0,    # NS mass
                      0,    #S mass
                      300,    # Fresh mass shoot
                      0,    # DLI (one pp per day!)
                      0,    # DLI lower
                      0,    # Average hourly cost
                      0,    # Avg PPFD during pp
                      0     # last u
                      ])   
    #x_ref = np.array(opt_config['x_ref'])
    Q_mat = np.diag([0, # NS mass                   0
                     0, # S mass                    1
                     0, # Fresh mass one shoot      2
                     0, # DLI uppert (one pp per day!)     3
                     0, # DLI lower bound           4
                     1, # Average hourly cost       5
                     0, # Avg PPFD during photoperiod  6
                     0
                     ])
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

        
        #x0l = x0
        #x0l[7] = 0
        #x0u = x0
        #x0u[7] = 0
        #ocp.constraints.lbx_0 = x0l
        #ocp.constraints.ubx_0 = x0u
        #ocp.constraints.idxbxe_0 = np.array([0,1,2,3,4,5,6,7])
        #ocp.constraints.idxbx_0 = np.array([0,1,2,3,4,5,6, 7])

        # Constraints on the intermediate stages
        #       DLI constraint: -100, 
        #ocp.constraints.lbx = np.array([-INF, -INF, -INF, -INF, -INF, -INF, -INF,-INF])
        #ocp.constraints.ubx = np.array([ INF, INF, INF, INF, INF, INF, INF, INF])
        #ocp.constraints.idxbx = np.array([0,1,2,3,4,5,6,7])
        
        # 10 days: 35, 40
        # 2 days: 

        ocp.constraints.lbx_e = np.array([5.5, -INF])
        ocp.constraints.ubx_e = np.array([200, INF])
        ocp.constraints.idxbx_e = np.array([2, 6])
        #
        #ocp.constraints.lbx = np.array([-INF , -INF])
        #ocp.constraints.ubx = np.array([INF, INF])
        #ocp.constraints.idxbx = np.array([2, 6])

        tol = 200
        #ocp.constraints.lh = np.array([ -INF])
        #ocp.constraints.uh = np.array([INF])
        #ocp.constraints.lh_0 = np.array([-INF])
        #ocp.constraints.uh_0 = np.array([INF])
        #ocp.constraints.lh_e = np.array([-INF, -INF])
        #ocp.constraints.uh_e = np.array([INF, INF])

    

    ocp.solver_options.qp_solver =  opt_config['qp_solver']
    ocp.solver_options.hessian_approx = opt_config['hessian_approx']
    ocp.solver_options.integrator_type = opt_config['integrator_type']
    ocp.solver_options.print_level = opt_config['print_level']
    ocp.solver_options.nlp_solver_type = opt_config['nlp_solver_type']
    #ocp.solver_options.sim_method_newton_iter = opt_config['sim_method_newton_iter']
    #ocp.solver_options.sim_method_num_stages = opt_config['sim_method_num_stages']
    #ocp.solver_options.sim_method_num_steps = opt_config['sim_method_num_steps']
    #ocp.solver_options.sim_method_newton_tol = opt_config['sim_method_newton_tol']

    
    #ocp.solver_options.nlp_solver_max_iter = opt_config['nlp_solver_max_iter']

    #ocp.solver_options.qp_solver_iter_max = opt_config["qp_solver_iter_max"]
    ocp.solver_options.nlp_solver_tol_stat = opt_config['nlp_solver_tol_stat']


    # set prediction horizon
    ocp.solver_options.tf = Tf
    json_file = 'mpc_optimization/acados_ocp.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = json_file)

    # create an integrator with the same settings as used in the OCP solver.
    #json_file = 'mpc_optimization/acados_sim.json'
    acados_integrator = AcadosSimSolver(ocp, json_file = json_file)

    return acados_ocp_solver, acados_integrator, ocp
