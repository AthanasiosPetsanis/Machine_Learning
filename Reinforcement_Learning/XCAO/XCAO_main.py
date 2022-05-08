# Code panel for X-CAO family algorthm in Python #

"""
 =======================================================================================
 This function acts as the main function of the XCAO CODE
 =======================================================================================
"""
import os
import sys
import gym
import numpy
import random
import scipy
import time
import DefineParameters
import XCAO_functions
from Pendulum import PendulumEnv



def XCAO():
    # Define the number of episodes
    nof_episodes = 10

    # Define the enviroment to use
    name_of_env = 'PendulumEnv'
    #env = gym.make(name_of_env)
    env = PendulumEnv()

    # Definition of parameters used in the problem
    Param = DefineParameters.Parameter_Definition(env)
    main_path = os.getcwd() + '\XCAO_results\Problem=' + name_of_env + '\\'
    if not os.path.isdir(main_path):
        os.mkdir(main_path)
    print("Directory '%s' created" % main_path)
    [P_matrix, Active_beta, symbolic_functions,random_selection] = Initialization(main_path, Param, nof_episodes)

    # Buffer Initialization
    Buffer_Cost = []
    Buffer_Matrix = []
    Buffer_ECost = []


    '''-----------------Experiment Loop-----------------------'''
    episode = 0
    while episode <= nof_episodes:
        State_History = []
        Beta_History = []
        Control_History = []
        Control_applied_History = []
        PCost_History = []
        ECost_History = []
        z_History = []
        Matrix_History = []
        current_folder = main_path + 'Episode=' + str(episode) + '\\'
        if not os.path.isdir(current_folder):
            os.mkdir(current_folder)


        iteration = 0
        done = 0
        while done == 0 and iteration < 15000:
            '''-----------------Decision Calculation-----------------------'''
            if iteration <= 0:
                previous_Control = Param['RBC'] * numpy.ones((Param['Output_Control'], 1))
                env.seed = random_selection[episode]
                States = env.reset()
            else:
                previous_Control = Control_History[iteration-1]
            [U_applied,U_Control,z_c,Active_beta] = Decision_Calculation(Param, States, previous_Control, P_matrix, main_path,iteration, symbolic_functions)
            print('Control is:' + str(U_applied))
            print('Conditions are:' + str(States))

            '''-----------------New State Calculation-----------------------'''
            [observation, reward, done, info] = env.step(U_applied)
            P_cost=reward
            States_New=observation

            '''--------------History and Buffer Update----------------------'''
            State_History.append(States)
            Control_History.append(U_Control)
            Control_applied_History.append(U_applied)
            PCost_History.append(P_cost)
            z_History.append(z_c)
            Beta_History.append(Active_beta)
            Matrix_History.append(P_matrix)
            States = States_New

            '''-----------------New Matrix Calculation---------------------'''
            if iteration>=1:
                mi = min(Param['Training_History'] - 1, iteration)
                comparison_pert = numpy.zeros(mi)
                cost_pert = []
                Pertubation = []
                for jj in range(Param['nof_Perturbations']):
                    P_pert = XCAO_functions.generate_perturbation(Param, P_matrix, iteration)
                    for ii in range(mi):
                        comparison_pert[ii] = XCAO_functions.Error_Calculation(z_History[iteration-ii], z_History[iteration-1-ii],PCost_History[iteration-ii],
                                                                                Matrix_History[iteration-ii],P_pert,iteration,current_folder,
                                                                                Param,Beta_History[iteration-ii])
                    cost_pert.append(sum(comparison_pert))
                    Pertubation.append(P_pert)

                cost_best = min(cost_pert)
                P_matrix = Pertubation[cost_pert.index(min(cost_pert))]
                ECost_History.append(cost_best)

            print("Iteration is = " + str(iteration))
            if iteration % 10 == 0:
                scipy.io.savemat(current_folder + '\History_Data.mat',{'z_History': z_History, 'P_Cost_History': PCost_History,'ECost_History': ECost_History,
                                                                       'Matrix_History': Matrix_History,'State_History':State_History,'Active_History':Beta_History,
                                                                       'Control_History':Control_History,'Control_applied_History':Control_applied_History})
            iteration = iteration + 1


        print("Episode is:" + str(episode))
        print("Cost of the episode is:" + str(sum(PCost_History)))

        '''-----------------Buffer_Update-----------------------'''
        [Buffer_Matrix,Buffer_ECost,Buffer_Cost]=XCAO_functions.Buffer_Update(episode, Param, PCost_History,ECost_History,Matrix_History,Buffer_Matrix, Buffer_ECost,Buffer_Cost)

        '''-----------------Selection of Matrix for new Experiment-----------------------'''
        P_matrix=Matrix_History[-1]

        episode = episode+1
        scipy.io.savemat(main_path + '\Buffer.mat',{'Buffer_Cost': Buffer_Cost,'Buffer_ECost': Buffer_ECost, 'Buffer_Matrix': Buffer_Matrix})





def Initialization(current_folder, Param, nof_episodes):
    # Initialization of the P_matrix
    [_, P_matrix] = XCAO_functions.genPDMrandomly(Param['e1'], Param['e2'], Param['L'], Param['dim_sub'])
    # Creation of the Symbolic Functions
    [GC, sigma] = XCAO_functions.beta_creation(Param['L'], Param['CHI_MAX'], Param['CHI_MIN'])
    Active_beta = 1
    symbolic_functions = XCAO_functions.symbolic_calculations(Param['nof_States'], Param['nof_Actions'],
                                                              Param['nof_Disturbances'],
                                                              Param['PredictDistHorizon'], Param['nof_Constraints'],
                                                              Param['L'],
                                                              Param['alpha'], Param['eta'], GC, sigma, Param['U_MIN'],
                                                              Param['U_MAX'], Param['Lambda'], current_folder)

    random_selection = random.sample(range(0, 1000), nof_episodes)

    return P_matrix, Active_beta, symbolic_functions, random_selection

def Decision_Calculation(Param, Conditions, u_prev, P_matrix, current_folder, counter,symbolic_functions):

    [GC, sigma] = XCAO_functions.beta_creation(Param['L'], Param['CHI_MAX'], Param['CHI_MIN'])
    beta_c = XCAO_functions.timestep_calc_beta(Param['L'], Conditions, GC, sigma)
    Active_beta = numpy. argmax(beta_c)
    ''' -------------------Feedback vector Creation - ------------------------------'''
    [x_c, xbar_c] = XCAO_functions.feedback_creation(Param, Conditions, u_prev, current_folder, counter)
    [z_c, z_ref] = XCAO_functions.timestep_calc_z(Param['L'], beta_c, xbar_c, Param)
    #Mx = XCAO_functions.timestep_calc_Mx(Param['L'], xbar_c, 0, x_c, beta_c, symbolic_functions)
    states = z_c[Active_beta*Param['dim_sub']:Active_beta*Param['dim_sub']+Param['dim_sub']]
    [B, Q, R] = XCAO_functions.Matrix_Initialization(Param)
    G = numpy.matmul(-B.T, P_matrix[:, :, Active_beta])
    U_Control = numpy.matmul(G, states)  # ficticious optimal control actions
    U_Control = (1-Param['pole']) * u_prev + Param['pole']* U_Control
    U_sat = Param['U_MAX'] * numpy.tanh(U_Control[Param['Output_Control']-1])
    V_c = numpy.matmul(numpy.matmul(states.T, P_matrix[:, :, Active_beta - 1]),states)
    return U_sat, U_Control ,z_c, Active_beta


if __name__ == "__main__":
    XCAO()