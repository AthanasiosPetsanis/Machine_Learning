import numpy as np
from numpy import asscalar
from numpy.random import normal as randn
from numpy.linalg import inv
import pickle
import os
import math
from math import pi, cos, sin, tanh
from random import randint
from NN import NeuralNetwork
import torch
from torch import sum, transpose, square, matmul, as_tensor
from torch import float64 as float32
import torch.optim as optim
from torch.autograd.functional import jacobian as jacob
import scipy.io
from time import time
from matplotlib.pyplot import plot, subplot, figure, title, grid, ylabel, xlabel, show


# Current Folder path
cur_folder = os.path.dirname(os.path.realpath(__file__))

nof_EPANAL = 1
for EPANAL in range(nof_EPANAL):
    print(EPANAL)
    rand_init_pos = 0
    duration = 5

    # tic
    start = time()

    # Load weights
    init_w = scipy.io.loadmat('INIT_WEIGHTS1.mat') 
    input_len, H1 = init_w['input_len'].item(), init_w['H1'].item()
    w1, w2 = init_w['w1'], init_w['w2']
    NN = NeuralNetwork(input_len, H1, w1, w2)

    # CONTROLLER PARAMETERS
    P = 0.01; inv_P = as_tensor([1/P], dtype=float32)
    Gamma = as_tensor(np.reshape([0,0,0,0,1],(5,1)), dtype=float32); Gamma_T = transpose(Gamma, 0, 1)
    Lambda = 0.8
    learn_mode = 1
    lr = 1e-5
    optimizer = optim.SGD(NN.parameters(), lr=lr)

    # Cart-Pole Parameters
    dt = .02
    g = 9.84
    cart_mass = 1
    pole_mass = 0.1
    total_mass = cart_mass + pole_mass
    length = 0.5
    pole_mass_length = pole_mass * length
    force = 10
    theta_threshold = 90
    neg_reward_threshold = 20
    neg_reward_threshold_rads = neg_reward_threshold*2*pi/360
    theta_threshold_rads = theta_threshold*2*pi/360
    x_threshold = 2.4

    # Load initial_pos
    initial_pos = scipy.io.loadmat('initial_pos.mat') 
    x0, x_dot0, th0, th_dot0 = initial_pos['x0'], initial_pos['x_dot0'], initial_pos['th0'], initial_pos['th_dot0']

    step = -1
    nof_epochs = 20
    nof_episodes = 1
    nof_steps = int(duration/dt) + 1
    total_steps = nof_epochs*nof_episodes*nof_steps

    R2REWARD = np.zeros(nof_epochs)
    CCOST = R2REWARD
    RREWARD = CCOST
    step_costs = np.zeros(nof_steps)
    ep_costs = np.zeros(nof_episodes)
    TRAIN_DATA = np.zeros((nof_episodes*nof_steps,12))
    history = {
        'Theta': np.zeros(total_steps),
        'X': np.zeros(total_steps),
        'Step_Costs': np.zeros(total_steps)
    }
    

    for epoch in range(nof_epochs):
        R2REWARD[epoch] = 0
        IITER = -1

        # Episodes
        for iter in range(nof_episodes):
            x = x0[iter][0]
            x_dot = x_dot0[iter][0]
            theta = th0[iter][0]
            theta_dot = th_dot0[iter][0]
            
            if rand_init_pos:
                rand_iter = randint(0,39)
                x, x_dot, theta, theta_dot = x0[rand_iter], x_dot0[rand_iter], th0[rand_iter], th_dot0[rand_iter]

            v = 0
            i = -1
            ep_costs[iter] = 0

            # Steps
            for t in np.arange(0,duration+dt,dt):
                reward = 0
                done = False
                step += 1
                IITER += 1
                i += 1

                # ------------------ Environment Box ------------------
                x_k = as_tensor([x, x_dot, theta, theta_dot, v], dtype=float32)
                nabla_NN = transpose(jacob(NN, x_k), 0, 1)
                u = (-0.5 * matmul(matmul(inv_P, Gamma_T), nabla_NN)).item()
                v_sat = force*tanh(v)#+0.1*randn()*v)
                #v_sat = force*tanh(v)

                temp = (v_sat + pole_mass_length * theta_dot**2 * sin(theta) ) / total_mass
                theta_acc = (g * sin(theta) - cos(theta) * temp)/ \
                            (length*(4/3 - pole_mass * cos(theta)**2/total_mass))
                x_acc = temp - pole_mass_length * theta_acc * cos(theta) / total_mass
                
                x = x + dt * x_dot
                x_dot = x_dot + dt * x_acc
                theta = theta + dt * theta_dot
                theta_dot = theta_dot + dt * theta_acc

                degrees_theta = theta*360/(2*pi)
                step_costs[i] = 0
                if theta > theta_threshold_rads or theta < -theta_threshold_rads or x > x_threshold or x < -x_threshold:
                    step_costs[i] = (theta-theta_threshold_rads)**2 + (x-x_threshold)**2
                R2REWARD[epoch] += step_costs[i]

                v = Lambda*v + (1-Lambda)*u
                # ------------------------------------------------------

                x_k_n = [x, x_dot, theta, theta_dot, v]
                history['Theta'][step], history['X'][step], history['Step_Costs'][step] = \
                    degrees_theta, x, step_costs[i]
                TRAIN_DATA[IITER,:] = x_k.tolist() + x_k_n + [u] + [step_costs[i]]
                ep_costs[iter] += step_costs[i]
        
        R2REWARD[epoch]
        optimizer.zero_grad() # grad = 0
        COST = 0
        REWARD = 0
        if learn_mode:
            for ii in range(IITER+1):
                vx = as_tensor(TRAIN_DATA[ii, 0:5], dtype=float32)
                vx_n = as_tensor(TRAIN_DATA[ii, 5:10], dtype=float32)
                vu = TRAIN_DATA[ii, 10]
                vr = TRAIN_DATA[ii, 11]
                nabla_NN_T = jacob(NN, vx, create_graph=True)
                nabla_NN = transpose(nabla_NN_T, 0, 1)
                calR = NN(vx) - NN(vx_n) - vr + 1/4 * matmul(matmul(matmul(matmul(nabla_NN_T,Gamma),inv_P),Gamma_T), nabla_NN) + matmul(nabla_NN_T, Gamma)*vu
                loss = 1/2*square(calR)#/calR.numel()
                loss.backward() # grad=grad+tt*at
                grad = NN.L2.weight.grad
                COST = COST + abs(calR)
                REWARD = REWARD + abs(vr)
            prev_weights = NN.L1.weight.tolist()
            optimizer.step() # w2=w2-lr*(grad)
            L2_weights = NN.L2.weight.tolist()
            post_weights = NN.L1.weight.tolist()
            assert prev_weights==post_weights, 'Weights of L1 changed'

            
        print(f'Epoch: {epoch},\n  Total calR for this epoch: {COST},\n  Total r for this Epoch: {REWARD}\n ')
        CCOST[epoch]=COST
        RREWARD[epoch]=REWARD

    epoch_min= np.where(RREWARD==np.amin(RREWARD))

# Print Results
last_epoch = total_steps - nof_steps + 1

figure(1)
subplot(2,1,1); plot(CCOST)
title('XCAO with gradient')
grid
ylabel('sum calR per epoch')
xlabel('Epoch')
subplot(2,1,2); plot(RREWARD)
grid
ylabel('Negative Reward')
xlabel('Epoch')

figure(2)
title('CartPole depiction')
subplot(3,1,1); plot(history['Theta'])
ylabel('Theta')
subplot(3,1,2); plot(history['X'])
ylabel('X')
subplot(3,1,3); plot(history['Step_Costs'])
ylabel('Cost')

figure(3)
title('CartPole depiction')
subplot(3,1,1); plot(history['Theta'][0:nof_steps])
plot(history['Theta'][last_epoch:step],'r')
grid
ylabel('Theta')

subplot(3,1,2); plot(history['X'][0:nof_steps])
plot(history['X'][last_epoch:step],'r')
grid
ylabel('X')

subplot(3,1,3); plot(history['Step_Costs'][0:nof_steps])
plot(history['Step_Costs'][last_epoch:step],'r')
grid
# axis([0 step/NO_OF_EPOCHS -0.1 1.1])
ylabel('Cost')

show()

# toc
end = time(); t_sec = end-start; mins = t_sec//60; secs = t_sec-mins*60
print(f"Training took {mins} minutes and {secs} seconds")