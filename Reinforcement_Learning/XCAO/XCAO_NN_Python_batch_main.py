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
from numpy.linalg import inv
import random
import scipy
import time
import DefineParameters
import XCAO_functions
from Network_Initialization import CreateNet
from torch.autograd.functional import jacobian as jacob
from torch.autograd import Variable
import torch
import torch.optim as optim
from Pendulum import PendulumEnv



def XCAO():
    nof_episodes = 50
    verbose = False

    # Define the enviroment to use
    name_of_env = 'PendulumEnv'
    #env = gym.make(name_of_env)
    env = PendulumEnv()

    # Definition of parameters used in the problem
    Param = DefineParameters.Parameter_Definition(env)
    lr = 1e-5

    dir_path = os.path.dirname(os.path.realpath(__file__))
    main_path = dir_path + '/XCAO_results/Problem=' + name_of_env + '/'
    if not os.path.isdir(main_path):
        os.makedirs(main_path)
        print("Directory '%s' created" % main_path)

    [NN_model,random_selection] = Initialization(main_path, Param, nof_episodes)
    
    # Buffer Initialization
    Buffer_Cost = []
    Buffer_Matrix = []
    Buffer_ECost = []

    '''-----------------Optimization Calculation---------------------'''
    optimizer = optim.SGD(NN_model.parameters(), lr=1e-4)
    # optimizer = optim.Adam(NN_model.parameters(), lr=0.0001)
    #lr=0.00001
    # optimizer.zero_grad()    # optimizer.zero_grad()


    '''-----------------Experiment Loop - number of NN updates/epochs-----------------------'''
    for epoch in range(400):
        # if verbose: print("=========Epoque is:============" + str(epoch))
        print("=========Epoque is:============" + str(epoch))
        episode = 0
        State_History = []
        Control_History = []
        Control_applied_History = []
        PCost_History = []
        ECost_History = []
        ECost_History_list = []
        z_History = []

        input_batch = torch.zeros([])
        target_batch = torch.zeros([])
        current_folder = main_path + 'Epoch=' + str(epoch) +  '//'
        if not os.path.isdir(current_folder):
            os.mkdir(current_folder)

        while episode < nof_episodes:
            iteration = 0
            done = 0
            while done == 0 and iteration < 100:
                '''-----------------Decision Calculation-----------------------'''
                if iteration <= 0:
                    previous_Control = Param['RBC'] * numpy.ones((Param['Output_Control'], 1))
                    #env.seed = random_selection[episode]
                    env.seed(episode)
                    States = env.reset()
                    #print(States)
                    #print('Conditions are:' + str(States))
                else:
                    previous_Control = Control_History[iteration-1]
                [U_applied,U_Control,z_c,Active_beta] = Decision_Calculation(Param, States, previous_Control, NN_model, main_path,iteration)
                #print('Conditions are:' + str(States))
                #print('Control is:' + str(U_applied))


                '''-----------------New State Calculation-----------------------'''
                [observation, reward, done, info] = env.step(U_applied)
                P_cost= - reward
                States_New=observation

                '''-----------------Error Calculation---------------------'''
                Error_t = XCAO_functions.Error_Calculation(numpy.append(States_New,U_Control,axis= 0),numpy.append(States,U_Control,axis= 0), U_Control, P_cost, NN_model,
                                                           current_folder, Param)

                '''--------------History and Buffer Update----------------------'''
                State_History.append(States)
                Control_History.append(U_Control)
                Control_applied_History.append(U_applied)
                PCost_History.append(P_cost)
                z_History.append(z_c)
                ECost_History.append(Error_t)
                ECost_History_list.append(Error_t.detach().numpy())

                States=States_New
                iteration=iteration+1


            if verbose: print("Episode is: " + str(episode))
            episode = episode+1

        scipy.io.savemat(current_folder + '/History_Data.mat',
                         {'z_History': z_History, 'P_Cost_History': PCost_History,
                          'ECost_History': ECost_History_list,
                          'State_History': State_History,
                          'Control_History': Control_History,
                          'Control_applied_History': Control_applied_History})


        '''-----------------Batch for feedforward ---------------------'''
        input_batch = torch.as_tensor(numpy.squeeze(numpy.array(z_History)), dtype=torch.float32)
        target_batch = torch.as_tensor(ECost_History,dtype=torch.float32)



        '''-----------------Optimization Calculation---------------------'''
        pretrained_dict = NN_model.state_dict()
        predictions = NN_model(input_batch)
        loss = MSE(predictions, target_batch) 

        # Make this a 'costum_step()' func?
        # with torch.no_grad():
        #     gradient_error = loss * jacob(loss,pretrained_dict['model.out.weight'])
        #     print(gradient_error.shape)
        # pretrained_dict['model.out.weight'] = pretrained_dict['model.out.weight'] - lr*gradient_error
        # NN_model.load_state_dict(pretrained_dict)


        # Make get_weights function? Shorcut-variables for NN_model.model.out.weight,bias ?
        # Weights before backprop
        pre_L2 = NN_model.model.out.weight.tolist()
        pre_L2.append(NN_model.model.out.bias.tolist())
        # Biases before backprop
        pre_L1 = NN_model.model.layers[0].weight.tolist()
        pre_L1.append(NN_model.model.layers[0].bias.tolist())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Weights after backprop
        post_L2 = NN_model.model.out.weight.tolist()
        post_L2.append(NN_model.model.out.bias.tolist())
        # Biases after backprop
        post_L1 = NN_model.model.layers[0].weight.tolist()
        post_L1.append(NN_model.model.layers[0].bias.tolist())

        # Show weight change    
        assert post_L2 != pre_L2, 'No change in targeted weights/biases after backprop'
        assert post_L1 == pre_L1, 'First layer weights/biases changed after backprop'

        print("Cost is:" + str(sum(PCost_History)))
        print('Error is:' + str((loss.detach().numpy())))

        Buffer_Cost.append(sum(PCost_History))
        Buffer_ECost.append(loss.detach().numpy())

        scipy.io.savemat(main_path + '/Evolution_Epoch.mat',
                         {'Buffer_Matrix': Buffer_Matrix, 'Buffer_ECost': Buffer_ECost,
                          'Buffer_Cost': Buffer_Cost})



def Initialization(current_folder, Param, nof_episodes):
    # Initialization of the P_matrix
    #[_, P_matrix] = XCAO_functions.genPDMrandomly(Param['e1'], Param['e2'], Param['L'], Param['dim_sub'])
    # Creation of the Symbolic Functions
    #[GC, sigma] = XCAO_functions.beta_creation(Param['L'], Param['CHI_MAX'], Param['CHI_MIN'])
    NN_model = CreateNet('fc', input_size=4, hid_neurons=100, hid_layers=1, output_size=1)

    # # Do not train first layer weights
    for param in NN_model.model.layers[0].parameters():
        param.requires_grad = False

    random_selection = [5,10,30,86,1110,110,666,234,987,2009]
    return NN_model,random_selection

def Decision_Calculation(Param, Conditions, u_prev, NN_model, current_folder, counter):

    [GC, sigma] = XCAO_functions.beta_creation(Param['L'], Param['CHI_MAX'], Param['CHI_MIN'])
    beta_c = XCAO_functions.timestep_calc_beta(Param['L'], Conditions, GC, sigma)
    Active_beta = numpy. argmax(beta_c)
    ''' -------------------Feedback vector Creation - ------------------------------'''
    [x_c, xbar_c] = XCAO_functions.feedback_creation(Param, Conditions, u_prev, current_folder, counter)
    [z_c, z_ref] = XCAO_functions.timestep_calc_z(Param['L'], beta_c, xbar_c, Param)
    states = z_c - z_ref
    [B, Q, R] = XCAO_functions.Matrix_Initialization(Param)
    #with torch.no_grad():
    NN_MX = jacob(NN_model, torch.Tensor(x_c)).T
    G = numpy.matmul(-1/2*inv(R),numpy.matmul(-B.T, NN_MX))
    U_Control = G.numpy().reshape(1,)
    U_sat = Param['U_MAX'] * numpy.tanh(U_Control[Param['Output_Control']-1])
    #V_c = numpy.matmul(numpy.matmul(states.T, P_matrix[:, :, Active_beta - 1]),states)
    return U_sat, U_Control ,z_c, Active_beta

def MSE(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

if __name__ == "__main__":
    XCAO()
