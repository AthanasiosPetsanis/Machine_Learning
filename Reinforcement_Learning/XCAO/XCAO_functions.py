'''
    =========================================================================
    This function contains all basic functions for XCAO implementation
    =========================================================================
'''

# import Modules
import DefineParameters
import Save_Load
from torch.autograd.functional import jacobian as jacob
import torch
# import Libraries
import os
import scipy.linalg as linear
import scipy.optimize as optimize
import numpy as np
import sympy
import operator
import math
import time
from scipy.special import erfinv

def genPDMrandomly(e1, e2, L, dim_subP):

    # np.random.seed(2)
    # P_ij=([],[],[])
    P_blk = np.empty((0, 0))
    A = np.diagflat(np.mean([e2, e1]) + np.random.uniform(-((e2 - e1) / 2), ((e2 - e1) / 2), [dim_subP, 1]))
    P_ij = np.empty([dim_subP, dim_subP, L])

    for i in range(L):
        Q = linear.orth(np.random.randn(dim_subP, dim_subP))
        PDM = np.dot(np.dot(Q.T, A), Q)
        P_ij[:, :, i] = PDM
        P_blk = linear.block_diag(P_blk, P_ij[:, :, i])
    '''
    # for i=1:L
    #	temp = randn(dim_subP);
    #	[U,ignore] = eig((temp+temp')/2);
    #	PDM = U*diag(abs(e1 + (e2-e1)*rand(dim_subP,1)))*U';
    #	P_ij(:,:,i) = PDM;
    #	P_blk = blkdiag(P_blk,P_ij(:,:,i));
    '''
    #P_blk=e1*np.eye(dim_subP)
    #P_ij[:,:,L-1]=P_blk
    return P_blk, P_ij


def beta_creation(L, y_max, y_min):
    '''
        NORMALISED BETA FUNCTION CALCULATION (GAUSSIAN DISTRIBUTION IMPLEMENTED)

        y_min,y_max: are the vectors that contain the min and max values of the states
        L: is the number of the mixing functions used for system approximation
    '''
    # --------------------------------------------------------------------------
    # norm calculation of state vectors for all time instances in order to
    # construct properly the the beta functions and find their characteristics
    # --------------------------------------------------------------------------
    sigma = (np.linalg.norm(y_max) - min([0, (np.linalg.norm(y_min))]))/ (6 * L)  # Guassian variance
    GC_temp = np.zeros([L])  # Gaussian Centers
    GC_temp[0] = min([0, (np.linalg.norm(y_min))]) + 3 * sigma
    for i in range(1, L):
        GC_temp[i] = GC_temp[0] + 6 * (i) * sigma
    return GC_temp, sigma


def timestep_calc_beta(L, Conditions, GC, sigma):
    # THIS FUNCTION CALCULATES BETA VALUES FOR ONE TIMESTEP - FOR A SINGLE GIVEN chi
    tilde_beta = np.zeros([L])
    #tilde_beta = sympy.Matrix(tilde_beta)
    temp_norms = 0
    x = np.matmul(Conditions.T, Conditions)
    #if type(Conditions) is sympy.matrices.dense.MutableDenseMatrix:
        #x = x
    #else:
    temp_norms = math.sqrt(x)
    A = 0
    B = 0
    s = 0
    sympy.NumberSymbol
    beta = np.zeros([1, L])
    # NoZones = len(chi)

    for i in range(L):
        #if type(temp_norms) is sympy.MatPow:
            #GC = sympy.Matrix(GC[i])
        # Gaussian values for every time instance of state vector
        A = math.exp(-((temp_norms - GC[i]) ** 2) / (2 * sigma ** 2))
        tilde_beta[i] = A

    s = sum(tilde_beta)
    #beta = 1
    beta = tilde_beta / s
    return beta


# ============THIS FUNCTION CALCULATES THE NUMERIC VALUES OF THE JACOBIAN MATRIX=======#
def timestep_calc_Mx(L, x_bar, x_bar_des, x, beta_multi, symbolic_functions):
    '''
    # THIS FUNCTION CALCULATES THE NUMERIC VALUES OF THE JACOBIAN MATRIX
    '''

    # load([current_folder,'Symbolic_function'],'fh_Jacobian_xbar','fh_Jacobian_SQRT_beta','fh_SIGMA')
    dim_x = len(x)
    dim_sub = len(x_bar)
    x = np.array(x)
    dim = dim_sub - 9

    fh1 = symbolic_functions[1]
    # fh1 = str2func(char(fh_Jacobian_xbar)) #https://mail.python.org/pipermail/python-list/2007-March/425058.html
    # Mx = np.zeros([dim_sub*L, dim_x])
    Mx = np.zeros([dim_sub, dim_x])

    for i in range(L):
        if L > 1:
            fh2 = symbolic_functions[2][i]
            # fh2 = str2func(char(fh_Jacobian_SQRT_beta[i,:]))
            Mx[i * dim_sub:(i + 1) * dim_sub, :] = (x_bar - x_bar_des) * fh2(x[0], x[1], x[2]) + np.sqrt(
                float(beta_multi[i])) * fh1(x[0], x[1], x[2])
        else:
            # Mx[i*dim_sub:(i+1)*dim_sub, :] = np.sqrt(float(beta_multi[i])) * fh1(x[0],x[1],x[2])
            s = "fh1("
            for ii in range(len(x) - 1):
                s += "x[" + str(ii) + "],"
            s += "x[" + str(len(x) - 1) + "])"
            ttt = (eval(s))
            # Mx[i * dim_sub:(i + 1) * dim_sub, :] = np.sqrt(float(beta_multi[i])) * fh1(x[0], x[1], x[2], x[3],
            # x[4],x[5], x[6])
            Mx[i * dim_sub:(i + 1) * dim_sub, :] = np.sqrt(float(beta_multi)) * ttt
            # x_bar=sympy.Matrix(x_bar)
        # Mx[i * dim_sub:(i + 1) * dim_sub, :] = np.sqrt(float(beta_multi[i])) * fh1(x)

    return Mx


def symbolic_calculations(n, m, num_of_dist, PredictDistHorizon, number_of_constraints, L, alpha, eta, GC,
                          sigma, U_min, U_max, Lambda, current_folder):
    JacobiansPath = current_folder + '\Jacobian' + '\\'
    if not os.path.isdir(JacobiansPath):
        os.makedirs(JacobiansPath)

    sym_flag = 1
    print('\nSymbolic function generation...')
    # symbolic creation of state
    chi = np.zeros((n, 1))
    chi = sympy.Matrix(chi)
    for i in range(0, n):
        chi[i] = sympy.var('chi' + str(i), real=True)

    # symbolic creation of unconstrained control inputs
    u_bar = np.zeros((m, 1))
    u_bar = sympy.Matrix(u_bar)
    for i in range(m):
        u_bar[i] = sympy.var('u_bar' + str(i), real=True)

    # symbolic creation of saturated control inputs
    #u = sigmoid(u_bar, Lambda, U_min, U_max)
    u=u_bar

    #beta = timestep_calc_beta(L, chi, GC, sigma)
    beta=0
    CONSTR = np.zeros((number_of_constraints, 1))
    CONSTR = sympy.Matrix(CONSTR)
    for i in range(number_of_constraints):
        CONSTR[i] = sympy.var('CONSTR' + str(i), real=True)

    # symbolic creation of unconstrained control inputs
    d = np.zeros((num_of_dist, 1))
    d = sympy.Matrix(d)
    for i in range(0, num_of_dist):
        d[i] = sympy.var('d' + str(i), real=True)

    # symbolic creation of unconstrained control inputs
    pd = np.zeros((PredictDistHorizon, 1))
    pd = sympy.Matrix(pd)
    for i in range(PredictDistHorizon):
        pd[i] = sympy.var('pd' + str(i), real=True)

    # 	#  --------------------------------------------
    # 	#   User defined symbolic constraint forms
    #	#	constraints (symbolic) of the form C(y) < 0
    # 	#  --------------------------------------------
    # 	if (number_of_constraints>0)
    # 		CONSTR(1:n) = -chi;                    # 0 <= x_i
    # 		CONSTR(n+1:2*n) = chi-X_max;           # x_i <= x_i,max
    # 		CONSTR(2*n+1:3*n+m) = -u_bar-U_min;    # 0 <= G_i and g_i,min <= g_i
    # 		for i=1:n
    # 		   idx=find(Row(i,:)==-1);
    # 		   CONSTR(225+i) = u_bar(i)-sum(u_bar(n+idx));  # G_i <= sum g_i (that receive r.o.w.)
    #
    # 		end
    # 		counter=1;
    # 		for i=1:Nr_junctions
    # 			#sum g <= C - L for every junction
    # 		  CONSTR(270+Nr_junctions+i) = sum(u_bar(counter:counter+Stages(i)))-Cycle-Losttime(i);
    # 			#sum g >= C - L for every junction
    # 		  CONSTR(286+Nr_junctions+i) = -sum(u_bar(counter:counter+Stages(i)))+Cycle+Losttime(i);
    # 		  counter=counter+Stages(i);
    # 		end
    # 	end
    # penality functions

    sigma = np.zeros(number_of_constraints)
    sigma = sympy.Matrix(sigma)
    for i in range(number_of_constraints):
        sigma[i] = np.exp(alpha * CONSTR[i] - eta)
    # augmented state vector x and augmented output vector y
    x_bar = np.concatenate((chi, u, d, pd, sigma), axis=0)
    x = np.concatenate((chi, u_bar), axis=None)
    #  -------------------------------------------
    #  Transformation of dynamical system equation
    #  -------------------------------------------

    #     Calculate M matrix symbolic parts
    # addpath(JacobiansPath)
    os.sys.path.append(JacobiansPath)
    #     flag = 0;
    #     while true
    # string = JacobiansPath + 'ConstraintsL' + str(L) + '_' + str(System)
    MM0 = sympy.lambdify(x, sigma)  # 0
    # fh_sigma = 'ConstraintsL' + str(L) + '_' + str(System)

    #         if number_of_constraints>0
    #             flag = checkSymbolics(fh_sigma);
    #         else
    #             flag = 1;
    #         end
    #         if flag
    #             break
    #         end
    #     end

    #     flag = 0;
    #     while true

    M = sympy.Matrix(x_bar).jacobian(x)  # 1
    # string = JacobiansPath + 'XbarJacobianL' + str(L) + '_' + str(System)
    MM1 = sympy.lambdify((x), M)  # 2
    # fh_Jacobian_xbar = 'XbarJacobianL' + str(L) + '_' + str(System)

    #         flag = checkSymbolics(fh_Jacobian_xbar);
    #         if flag
    #             break
    #         end
    #     end

    #     flag = 0;
    #     while true

    fh_Jacobian_SQRT_beta = []
    MM2 = []

    for i in range(L):
        M = sympy.Matrix([sympy.sqrt(beta)]).jacobian(x)
        # string = JacobiansPath + 'BetaSQRTJacobian' + str(i) + str(L) + '_' + str(System)
        MM2_temp = sympy.lambdify(x, M)  # 4
        # fh_Jacobian_SQRT_beta.append('BetaSQRTJacobian' + str(i) + str(L) + '_' + str(System))
        MM2.append(MM2_temp)

    #         flag = checkSymbolics(fh_Jacobian_SQRT_beta);
    #         if flag
    #             break;
    #         end
    symbolic_functions = [MM0, MM1, MM2]
    Save_Load.save(symbolic_functions, JacobiansPath, 'Symbolic_functions', 'pickle')

    return symbolic_functions


def sigmoid(u_bar, Lambda, min_value, max_value):
    '''
        #SIGMOID Compute sigmoid function
    '''

    m = len(u_bar)
    # print('\n ssssss', type(u_bar), '\n', type(Lambda), '\n', type(min_value), '\n', type(max_value), '\n', sym_flag)
    u = np.zeros([m, 1])
    Lambda = Lambda.flatten()


    for i in range(m):
        a2 = (max_value[i] + min_value[i]) / 2
        b2 = (max_value[i] - min_value[i]) / 2

        Lam = Lambda[i]

        aa = (u_bar[i] - a2)

        tt = (Lam * aa / b2)
        # print(i,a2,b2)

        k = np.shape(tt)
        u[i] = a2 + b2 * np.tanh(tt[0])
    return u


def feedback_creation(Param, Conditions, u_prev, current_folder, counter):
    if Param['nof_States'] > 0:
        chi_c = Conditions
    else:
        chi_c = []

    if Param['nof_Actions'] == 0:
        u_c = []
    else:
        u_c = u_prev

    if Param['nof_Disturbances'] == 0:
        dist_c = []
        predict_dist_c = []
    x_c = np.concatenate((dist_c, predict_dist_c, chi_c, u_c), axis=None)
    xbar_c = np.concatenate((dist_c, predict_dist_c, chi_c, u_c), axis=None)
    return x_c, xbar_c


# ========THIS FUNCTION CALCULATES z VALUES FOR ONE TIMESTEP - FOR A SINGLE GIVEN x_bar=======#
def timestep_calc_z(L, beta_multi, x_bar, Param):
    dim_sub = len(x_bar)
    z = np.zeros([dim_sub * L, 1])
    z_ref = np.zeros([dim_sub * L, 1])
    for i in range(L):
        z[i * dim_sub:(i + 1) * dim_sub, :] = np.transpose((sympy.sqrt(beta_multi[i]) * x_bar)[np.newaxis])
    return z, z_ref


def Buffer_Update(Iteration, Param, PCost_History,ECost_History,Matrix_History,Buffer_Matrix, Buffer_ECost,Buffer_Cost):

    #if Iteration <= buf_size-1:

    Buffer_Matrix.append(Matrix_History)
    Buffer_Cost.append(PCost_History)
    Buffer_ECost.append(ECost_History)
    '''else:
        end1 = np.shape(Buffer_States)
        end = end1[0]
        Buffer_States[0:buf_size - 1, :] = Buffer_States[1: end]
        Buffer_States[buf_size - 1, :] = Conditions.T

        Buffer_Control[0: buf_size - 1] = Buffer_Control[1: end]
        Buffer_Control[buf_size - 1, :] = U_Control.T

        Buffer_Cost[0: buf_size - 1] = Buffer_Cost[1: end]
        Buffer_Cost[buf_size - 1, :] = P_Cost.T

        Buffer_Matrix[0: buf_size - 1] = Buffer_Matrix[1: end]
        Buffer_Matrix.append(P_matrix)

        Buffer_z[0:buf_size - 1, :] = Buffer_z[1: end]
        Buffer_z[buf_size - 1, :] = z_c'''

    return Buffer_Matrix,Buffer_ECost,Buffer_Cost


def Error_Calculation(z_new,z,u_control_old, Cost, NN_model, current_folder,Param):

    x_new = z_new.reshape(4,)
    x_old = z.reshape(4,)
    P_Kostos = Cost
    [B_curr, Q, R] = Matrix_Initialization(Param)
    #G = np.matmul(np.matmul(np.matmul(np.matmul(M.T, B_curr), RR), B_curr.T), M)
    RR = np.linalg.inv(R)


    '''x1_grad = NN_model(torch.Tensor(x_old))
    with torch.no_grad():
        x1_nograd = NN_model(torch.Tensor(x_new))
    x1 = x1_grad - x1_nograd - Cost
    x2 = np.matmul(np.matmul(NN_MX,B_curr*RR*B_curr.T),NN_MX.T)
    x3 = np.matmul(NN_MX,B_curr*u_control_old)'''
    # xold_grad = NN_model(torch.Tensor(x_old))


    NN_MX = jacob(NN_model, torch.Tensor(x_old)).T
    xnew_nograd = NN_model(torch.Tensor(x_new))

    B_curr=torch.as_tensor(B_curr, dtype=torch.float32)
    RR=torch.as_tensor(RR, dtype=torch.float32)
    u_control_old=torch.as_tensor(u_control_old, dtype=torch.float32)


    x1 = Cost + xnew_nograd

    x2 = -1/4  * RR * torch.matmul(torch.matmul(torch.matmul(NN_MX.T,B_curr),B_curr.T),NN_MX)

    #x3 = 1/2 * RR * torch.matmul(torch.matmul(NN_MX.T,B_curr) , u_control_old)

    x3 = - torch.matmul(torch.matmul(NN_MX.T, B_curr), u_control_old)

    R = x1 + x2 + x3



    return R


def generate_perturbation(Param, P_compl, iter):
    dim_sub = np.shape(P_compl)
    dim_sub = dim_sub[0]
    L = Param['L']
    pert_P=np.zeros([dim_sub,dim_sub,L])

    for ii in range(L):
        samples = np.random.rand(dim_sub, 1)
        temp = np.sqrt(2) * erfinv(2 * samples - 1)
        ignore, U = np.linalg.eig((temp + temp.T) / 2)
        di = abs(Param['e1'] + (Param['e2'] - Param['e1']) * np.random.random([dim_sub, 1]))
        diag = np.diagflat(di)
        delta_P = np.matmul(np.matmul(U, diag), U.T)
        a = Param['PerturbStep']
        pert_P[:, :, ii] = (1 - a) * P_compl[:, :, ii] + a * delta_P
        #pert_P[:, :, ii] = P_compl[:, :, ii] + a * delta_P
        #pert_P[:, :, ii] = delta_P
    return pert_P


def Matrix_Initialization(Param):

        # B = np.concatenate((np.zeros([Param['nof_States'], Param['nof_Actions']]), np.eye(Param['nof_Actions'])),
        # axis=0)
        #B = np.concatenate((np.ones([Param['nof_States']-Param['Output_Control'],Param['Output_Control']]), np.eye(Param['Output_Control'])),axis=0)
        B = np.concatenate((np.zeros([Param['nof_States'], Param['Output_Control']]), np.eye(Param['Output_Control'])),axis = 0)
        Q = np.eye(Param['Output_States'])
        R = np.eye(Param['Output_Control'])*0.01

        return B, Q, R