'''
    =========================================================================
    This function defines and saves useful parameters for PCAO implementation
    =========================================================================
'''

#import libraries
import numpy as np
    
print('\n Defining experiment and system parameters...')

def Parameter_Definition(env):
    # XCAO parameters
    # ====================================================================\
    e1 = 1 * 10 ** (-1)  # 4*10^(-7) # lower bounds of P matrix elements
    e2 = 3 * 10 ** (0)  # 7 upper bounds of P matrix elements
    nof_Perturbations = 20  # 250 number of random perturbations - tests
    Training_History = 100  # set of history_data used for training
    pole = 0.2  # stabilization slow pole in fictitious control integration
    GlobalCapBuffer = 4  # 4global performance history points to be incorporated in the estimator's regressor vector
    w_norm = 10  # normalisation bound i.e. [ - w_norm, + w_norm ] for estimator training data
    PerturbStep = 0.2  # 0.1 Perturbation searching step
    eta = 1  # exp(-alpha*(CONST(i) - eta) functions
    alpha = 0.1


    # System-Loop parameters
    # ===================================================================================
    observation = env.observation_space.sample()
    action = env.action_space.sample()
    t_u = 1
    t_x = 1
    RBC = 0
    Output_States = len(observation)
    Output_Control = len(action)

    # Parameters for state vector
    # ===================================================================================
    nof_Systems = 1
    L = 1  # number of controller mixing functions
    nof_States = Output_States  # state variables number
    nof_Actions = Output_Control  # 5        # control variables number
    nof_Disturbances = 0  # 3             # disturbance measurements # Temperature, humidity, solar radiation
    PredictDistHorizon = 2 * nof_Disturbances  # distrurbance prediction horizon hours
    nof_Constraints = 0
    dt = 60*60
    dim_sub = nof_States + nof_Actions

    # maximum/minimum values of control/state variables
    # ===================================================================================
    U_MIN = env.action_space.low  # respective minimum values of control variables
    U_MAX = env.action_space.high # respective maximum values of control variables

    CHI_MIN = env.observation_space.low # respective minimum values of state variables [ Temperatures and Humidities ]
    CHI_MAX = env.observation_space.high

    Lambda = 1 * np.ones((1, Output_Control))  # saturation function slope factor

    Param = {'e1': e1, 'e2': e2, 'nof_Perturbations': nof_Perturbations, 'pole': pole,
             'GlobalCapBuffer': GlobalCapBuffer, 'w_norm': w_norm, 'PerturbStep': PerturbStep,
             'eta': eta, 'Training_History':Training_History,'alpha':alpha,
             'nof_Systems': nof_Systems, 'L': L, 'nof_States': nof_States, 'nof_Actions': nof_Actions,
             'nof_Disturbances': nof_Disturbances,
             'PredictDistHorizon': PredictDistHorizon, 'dim_sub': dim_sub, 'nof_Constraints': nof_Constraints,
             'dt': dt, 'U_MIN': U_MIN, 'U_MAX': U_MAX, 'CHI_MIN': CHI_MIN, 'CHI_MAX': CHI_MAX, 'Lambda': Lambda,
              't_u': t_u, 't_x': t_x, 'RBC': RBC,
             'Output_States': Output_States, 'Output_Control': Output_Control}


    return Param