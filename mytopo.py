#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 02:28:06 2022

@author: hak
"""

import numpy as np
import torch
from ATCNN_model import normalization, to_binary, switch, translate, mapping
import math
import pandas as pd
from numpy.polynomial import Polynomial as P
from pymobility.models.mobility import random_walk
import random
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import tensorflow as tf
import imageio


REMOTE_CONTROLLER_IP = '127.0.0.1'

class Net():
    def __init__(self, AP_num, UE_num):     
        self.AP_num = AP_num
        self.UE_num = UE_num
        self.B = 20 # unit: Mbps
        if AP_num == 5:
            self.X_length = 5 # room size
            self.Y_length = 5 # room size
            self.Z_height = 3 # room height
            self.AP_position = [[2.5, 2.5, 0.5], [1.25, 1.25, 3], [3.75, 1.25, 3], [1.25, 3.75, 3], [3.75, 3.75, 3]]
        elif AP_num == 10:
            self.X_length = 9 # room size
            self.Y_length = 9 # room size
            self.Z_height = 3 # room height
            self.AP_position = [[4.5, 4.5, 0.5], [1.5, 1.5, 3], [4.5, 1.5, 3], [7.5, 1.5, 3], 
                   [1.5, 4.5, 3], [4.5, 4.5, 3], [7.5, 4.5, 3],
                   [1.5, 7.5, 3], [4.5, 7.5, 3], [7.5, 7.5, 3]]
        else: # AP number of 17, including 16 LiFi AP
            self.X_length = 10 # room size
            self.Y_length = 10 # room size
            self.Z_height = 3 # room height
            # self.AP_position =  [[8, 8, 0.5], [2, 2, 3], [6, 2, 3], [10, 2, 3], [14, 2, 3],
                          # [2, 6, 3], [6, 6, 3], [10, 6, 3], [14, 6, 3],
                          # [2, 10, 3], [6, 10, 3], [10, 10, 3], [14, 10, 3],
                          # [2, 14, 3], [6, 14, 3], [10, 14, 3], [14, 14, 3]]
            self.AP_position =  [[5, 5, 0.5], [1.25, 1.25, 3], [3.75, 1.25, 3], [6.25, 1.25, 3], [8.75, 1.25, 3],
                          [1.25, 3.75, 3], [3.75, 3.75, 3], [6.25, 3.75, 3], [8.75, 3.75, 3],
                          [1.25, 6.25, 3], [3.75, 6.25, 3], [6.25, 6.25, 3], [8.75, 6.25, 3],
                          [1.25, 8.75, 3], [3.75, 8.75, 3], [6.25, 8.75, 3], [8.75, 8.75, 3]]
            

        
    def mobility_trigger(self, time_slot, X_coordinate, Y_coordinate, Saved_SNR_list):
        # update position
        self.UE_position = self.recorded_trace[time_slot]
        SNR_matrix = []
        Capacity = []
        for i in range(self.UE_num):
            # find pre-defined SNR values
            X_value = min(X_coordinate, key=lambda x:abs(x - self.UE_position[i][0]))
            Y_value = min(Y_coordinate, key=lambda x:abs(x - self.UE_position[i][1]))
            
            X_index = X_coordinate.index(X_value)
            Y_index = Y_coordinate.index(Y_value)
            
            snr_index = len(Y_coordinate)*X_index + Y_index # <------ Right here
            snr_list = Saved_SNR_list[snr_index]
            snr_list = np.clip(snr_list, -20, 70).tolist()
            Capacity_vector = [self.B*math.log2(1 + 10**(j/10)) for j in snr_list] # should for WiFi, not for LiFi
            SNR_matrix.append(snr_list) 
            Capacity.append(Capacity_vector)
        self.SNR_matrix = SNR_matrix
        self.Capacity = Capacity        
    
    def load_balancing(self, Trained_Net, M):
        # load ATCNN model, correct now
        target_list = []
        user_dim = self.AP_num + 1
        R_list = self.R_requirement
        # get target
        SNR = self.SNR_matrix
        for i in range(self.UE_num):
            target = SNR[i][0:self.AP_num]
            target.append(R_list[i]*1e6)
            target_list.append(target)
        # get condition    
        condition = np.reshape(target_list, ((self.AP_num+1)*self.UE_num, 1)).tolist()
        condition = sum(condition, []) # flatten the list
        predicted_output = []
        tic = time.time()
        for i in range(self.UE_num):
            raw_condition = torch.tensor(condition)
            condition_now = switch(raw_condition, i, self.AP_num+1) 
            condition_now = torch.tensor([condition_now])
            mirroring_condition = mapping(M-1, [self.UE_num], condition_now.tolist(), self.AP_num)
            nor_mirroring_condition = normalization(M, self.AP_num, mirroring_condition, 60, -20, 1000) # nomalization is correct
            nor_mirroring_condition = torch.tensor(nor_mirroring_condition).to(torch.float32)
            Target = nor_mirroring_condition[..., 0:user_dim]    
            Condition = nor_mirroring_condition[..., 0:]
            
            output = Trained_Net.forward(Target, Condition) 
            binary_output = to_binary(self.AP_num, output.tolist())
            predicted_output.append(binary_output)
        toc = time.time()
        runtime = toc - tic
        self.X_iu = translate(predicted_output)
        return runtime

    def load_balancing_SSS(self):
        # do SSS LB only for target UE
        # get target
        tic = time.time()
        X_iu = []
        for i in range(self.UE_num):
            SNR_list = self.SNR_matrix[i]
            max_SNR = max(SNR_list)
            index = SNR_list.index(max_SNR)
            X_iu.append(index + 1)
        toc = time.time()
        self.X_iu = X_iu
        return (toc-tic)

    def load_balancing_GT(self):
        #######
        self.load_balancing_SSS() # use SSS as initial X_iu
        # calculate initial satisfaction list
        Rho_iu = PA_optimization(self.B, self.AP_num, self.UE_num, self.X_iu, self.R_requirement, self.Capacity, 0)
        Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num
        Satisfaction_vector = []
        for ii in range(self.UE_num):
            list1 = self.Capacity[ii]
            list2 = Rho_transposed[ii]
            sat_now = min(sum(list(np.multiply(list1, list2)))/self.R_requirement[ii], 1)      
            Satisfaction_vector.append(sat_now)
        #######
        N_f = 5  # number of AP candidates for each UE
        count = 0
        mode = 0
        payoff_vector = [0]
        N = self.UE_num
        X_iu_old = self.X_iu
        
        
        while mode <= N:
            estimated_payoff = [0]*N_f
            mutation_probability = np.zeros(self.UE_num)
            aver_payoff = sum(Satisfaction_vector)/self.UE_num
            
            for i in range(self.UE_num):
                if Satisfaction_vector[i] < aver_payoff:
                    mutation_probability[i] = 1 - Satisfaction_vector[i]/aver_payoff
                else:
                    mutation_probability[i] = 0
                
                x = np.random.rand(1)
                # apply mutation rule here
                if x < mutation_probability[i]:
                    old_AP = X_iu_old[i]
                    
                    # find 5 AP candidates for mutation UE
                    SNR_list = self.SNR_matrix[i]
                    AP_index = sorted(range(len(SNR_list)), key=lambda i: SNR_list[i], reverse=True)[:N_f]
                    
                    for j in range(N_f):
                        X_iu_old[i] = AP_index[j] + 1 # update X_iu
                        Rho_iu = PA_optimization(self.B, self.AP_num, self.UE_num, X_iu_old, self.R_requirement, self.Capacity, 0)
                        Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num
                        
                        sat_list = []
                        for ii in range(self.UE_num):
                            list1 = self.Capacity[ii]
                            list2 = Rho_transposed[ii]
                            sat_now = min(sum(list(np.multiply(list1, list2)))/self.R_requirement[ii], 1)      
                            sat_list.append(sat_now)

                        estimated_payoff[j] = sum(sat_list)/self.UE_num
                        
                    AP_muted = [AP_index[ii] for ii in range(N_f) if estimated_payoff[ii] == max(estimated_payoff)]
                    
                    if len(AP_muted) > 1:
                        SNR_set = [self.SNR_matrix[i][k] for k in AP_muted]
                        index = SNR_set.index(max(SNR_set))
                        AP_muted_chosen = AP_muted[index]
                        X_iu_old[i] = AP_muted_chosen + 1 # update X_iu here
                    else:
                        X_iu_old[i] = AP_muted[0] + 1 # update X_iu here
                    
                    AP_mutated = X_iu_old[i]
                    
                    if AP_mutated != old_AP:
                        Rho_iu = PA_optimization(self.B, self.AP_num, self.UE_num, X_iu_old, self.R_requirement, self.Capacity, 0)
                        Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num
                        
                        Satisfaction_vector = []
                        for ii in range(self.UE_num):
                            list1 = self.Capacity[ii]
                            list2 = Rho_transposed[ii]
                            sat_now = min(sum(list(np.multiply(list1, list2)))/self.R_requirement[ii], 1)      
                            Satisfaction_vector.append(sat_now)

                        aver_payoff = sum(Satisfaction_vector)/self.UE_num
                        
                        if aver_payoff > payoff_vector[-1]:
                            mode = 0
                            payoff_vector.append(aver_payoff)
                        else:
                            mode += 1
                    else:
                        mode += 1
                else:
                    # no mutation action for UE j
                    payoff_vector.append(payoff_vector[-1])
                    
                if mode > N:
                    break
                count += 1
            if aver_payoff >= 0.99:
                break
        self.X_iu = X_iu_old
        results = [aver_payoff, count, payoff_vector]
        return results
                   

def distance(A, B):
    dis = math.sqrt(( float(A[0]) - float(B[0]))**2 + (float(A[1]) - float(B[1]))**2)    
    return dis

def angle2(v1, v2):
    x=np.array(v1)
    y=np.array(v2)
    module_x=np.sqrt(x.dot(x))
    module_y=np.sqrt(y.dot(y))
    dot_value=x.dot(y)
    cos_theta=dot_value/(module_x*module_y)
    angle_radian=np.arccos(cos_theta)
    # angle_value=angle_radian*180/np.pi
    return angle_radian

def PA_optimization(B, AP_num, UE_num, X_iu_list, R_required, Capacity, opt_mode):
    X_iu = np.zeros((AP_num, UE_num), dtype=int)
    for i in range(UE_num):
        X_iu[X_iu_list[i]-1, i] = 1

    Rho_iu = np.zeros((AP_num, UE_num))
    for i in range(AP_num):
        connected_UE = np.where(X_iu[i,:] == 1)[0]
        if connected_UE.size != 0:
            if opt_mode == 1:
                A = np.ones((1, connected_UE.size))
                b = 1
                lb = np.zeros((connected_UE.size,))
                ub = np.ones((connected_UE.size,))
                X0 = (lb+ub)/(connected_UE.size+1)
                options = {"maxiter": 10000, "ftol": 1e-10}
                res = minimize(object_function, X0, args=(Capacity[i,connected_UE],R_required[connected_UE]), method="SLSQP", bounds=list(zip(lb,ub)), constraints={"type": "eq", "fun": lambda x: np.dot(A, x) - b}, options=options)
                X = res.x
                Rho_iu[i, connected_UE] = X
            else:
                Rho_iu[i, connected_UE] = np.ones((connected_UE.size,)) / connected_UE.size

    return Rho_iu

def object_function(X, Capacity, R_required):
    cost = 0
    for i in range(Capacity.size):
        cost += np.log2(min(Capacity[i]*X[i]/R_required[i], 1))
    cost = -cost
    return cost

# for calculate average throughput using dynamic UTI and delay
def start_simualtion(net, eng, Trained_Net, M, opt_mode, UTI_mode, LB_mode, Thr_mode, runtime_mode=None):
    ###### Ideal Benchmark with 10ms UTI ###############################
    # Thr_mode: Target or Average
    # update env setup for HLWNets
    UTI_thr_list = []
    if runtime_mode == None:
        runtime_ATCNN = 0
        runtime_UTI = 0
        runtime_SSS = 0
        runtime_GT = 0
    else:
        runtime_ATCNN = 0.002
        runtime_UTI = 0.000164
        runtime_SSS = 0.000042
        # UE number from 10-100
        runtime_GT_list = [0.0151,0.0173,0.0194,0.0236,0.0257,0.029,0.0312,0.0331,0.0438,0.0524,0.0534,0.0634,0.0649,0.0640,0.0678,0.0782,0.0873,0.0987,0.109,0.110,
            0.120,0.124,0.129,0.145,0.161,0.164,0.179,0.185,0.188,0.208,0.212,0.231,0.226,0.253,0.321,0.289,0.285,0.329,0.361,0.352,0.451,
            0.473,0.419,0.518,0.489,0.463,0.473,0.475,0.550,0.560,0.636,0.654,0.601,0.636,0.681,0.709,0.636,0.672,0.728,0.760,0.730,0.761,0.766,0.927,0.901,1.01,
            1.098,0.928,1.068,1.015,1.129,1.146,1.242,1.072,1.183,1.121,1.290,1.397,1.387,1.420,1.456,1.470,1.386,1.412,1.583,1.638,1.578,1.600,1.751,1.844,1.958] 
        runtime_GT = runtime_GT_list[net.UE_num-10] # 
    
    tf.keras.utils.disable_interactive_logging()
    # load trained UTI models in advance
    net.loaded_model_Type4 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type4.h5')
    net.loaded_model_Type1 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type1.h5')
    net.loaded_model_Type3 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type3.h5')
    net.loaded_model_Type2 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type2.h5')
    
    UTI_update_instances = []
    LB_update_instances = []
    for n in range(net.waypoints_num-1):
        # mobility model
        net.mobility_trigger(n, net.X_list_new, net.Y_list_new, net.Saved_SNR_list) # update SNRs and Capacity  
        
        # done, average throughput for ATCNN@10ms UTI
        if UTI_mode == '10ms':  
            delay = runtime_ATCNN
            delay_time = int(delay/0.01)
            net.load_balancing(Trained_Net, M)
            Rho_iu = PA_optimization(net.B, net.AP_num, net.UE_num, net.X_iu, net.R_requirement, net.Capacity, opt_mode)
            Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num   
            ########### IDeal throughput: For target UE ###########    
            if n < delay_time:
                UTI_thr_now = 0 # save throughput
            else:
                if Thr_mode == 'Target':
                    for i in range(1):
                        list1 = net.Capacity[i]
                        list2 = Rho_transposed[i]
                        UTI_thr_now = min(sum(list(np.multiply(list1, list2))), net.R_requirement[i])
                else: # for average throughput
                    UTI_thr_now = []
                    for i in range(net.UE_num):
                        list1 = net.Capacity[i]
                        list2 = Rho_transposed[i]
                        UTI_thr_now.append(min(sum(list(np.multiply(list1, list2))), net.R_requirement[i]))
                    UTI_thr_now = sum(UTI_thr_now)/len(UTI_thr_now)
                        
        ################# done, for average throughput for MS-ATCNN ####################
        elif UTI_mode == 'Dynamic':      
            if n == 0:
                net.load_balancing(Trained_Net, M)
                X_iu_old = net.X_iu # initial result, first update at t0
                sum_UTI = [0]*net.UE_num # sum_UTI is the time axis
                Delta_time_list = []
                for i in range(net.UE_num):
                    UTI = call_function(net, n, X_iu_old, i) # get UTIs for all UEs at t0
                    Delta_time_list.append(int(UTI/0.01))
                delay = runtime_ATCNN + runtime_UTI # runtime time to delay
                delay_time = int(delay/0.01)  
                UTI_update_instances.append(n) # record time instants when target UE needs to call ATCNN
            
            for j in range(net.UE_num):
                Delta_time = Delta_time_list[j] # UTI time for j-th UE
                if n == (sum_UTI[j] + Delta_time): # predict new UTI for j-th UE
                    sum_UTI[j] = sum_UTI[j] + Delta_time
                    UTI = call_function(net, n, X_iu_old, j)
                    Delta_time_list[j] = int(UTI/0.01) # update UTI for j-th UE
                    if j == 0: # only record for target UE to plot
                        UTI_update_instances.append(n)
            
            for j in range(net.UE_num):       
                if n == sum_UTI[j] + delay_time:
                    net.load_balancing(Trained_Net, M)
                    X_iu_old[j] = net.X_iu[j] # after delay of runtime, only update j-th UE
                    if j == 0: # only record for target UE to plot
                        UTI_update_instances.append(n)

            if n < delay_time:
                UTI_thr_now = 0 # save throughput
            else:
                Rho_iu = PA_optimization(net.B, net.AP_num, net.UE_num, X_iu_old, net.R_requirement, net.Capacity, opt_mode) 
                Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num
                #
                if Thr_mode == 'Target':
                    for i in range(1):
                        list1 = net.Capacity[i]
                        list2 = Rho_transposed[i]
                        UTI_thr_now = min(sum(list(np.multiply(list1, list2))), net.R_requirement[i])
                else: # for average throughput
                    UTI_thr_now = []
                    for i in range(net.UE_num):
                        list1 = net.Capacity[i]
                        list2 = Rho_transposed[i]
                        UTI_thr_now.append(min(sum(list(np.multiply(list1, list2))), net.R_requirement[i]))
                    UTI_thr_now = sum(UTI_thr_now)/len(UTI_thr_now)
                    if n%100 == 0:
                        print('*************** time index n ******************* is:', n)
        
        # done average throughput for Fixed UTI    
        else: # fixed UTI mode
            if LB_mode == 'SSS':
                delay = runtime_SSS
            elif LB_mode == 'GT':
                delay = runtime_GT
            else: # for ATCNN with fixed UTI
                delay = runtime_ATCNN
            delay_time = int(delay/0.01)
            
            if UTI_mode == '200ms':
                UTI = 0.2
            elif UTI_mode == '1000ms':
                UTI = 1 
            elif UTI_mode == '2000ms':
                UTI = 2
            else:
                if net.Aver_Velocity == 1:
                    UTI = 1.014 # Average UTI for SSS and GT baselines
                elif net.Aver_Velocity == 1.5:
                    UTI = 0.8273 # Average UTI for SSS and GT baselines
                elif net.Aver_Velocity == 2:   
                    UTI = 0.7415 # Average UTI for SSS and GT baselines
                elif net.Aver_Velocity == 2.5:
                    UTI = 0.6384 # Average UTI for SSS and GT baselines
                elif net.Aver_Velocity == 3:
                    UTI = 0.6263
                elif net.Aver_Velocity == 4:
                    UTI = 0.4896
                else: # 5m/s
                    UTI = 0.4427 # Average UTI for SSS and GT baselines
            
            Delta_time = int(UTI/0.01)  
            
            if (n - delay_time) % Delta_time == 0:
                net.mobility_trigger(n - delay_time, net.X_list_new, net.Y_list_new, net.Saved_SNR_list) # update SNRs and Capacity  
                if LB_mode == 'SSS':
                    net.load_balancing_SSS()
                    UTI_update_instances.append(n - delay_time)
                    LB_update_instances.append(n)
                elif LB_mode == 'GT':
                    net.load_balancing_GT()
                    UTI_update_instances.append(n - delay_time)
                    LB_update_instances.append(n)
                else: 
                    net.load_balancing(Trained_Net, M) # for ATCNN with fixed UTI
                X_iu_old = net.X_iu
                
            if n >= delay_time:
                Rho_iu = PA_optimization(net.B, net.AP_num, net.UE_num, X_iu_old, net.R_requirement, net.Capacity, opt_mode)
                Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num
                # 
                if Thr_mode == 'Target':
                    for i in range(1):
                        list1 = net.Capacity[i]
                        list2 = Rho_transposed[i]
                        UTI_thr_now = min(sum(list(np.multiply(list1, list2))), net.R_requirement[i])
                else: # for average throughput
                    UTI_thr_now = []
                    for i in range(net.UE_num):
                        list1 = net.Capacity[i]
                        list2 = Rho_transposed[i]
                        UTI_thr_now.append(min(sum(list(np.multiply(list1, list2))), net.R_requirement[i]))
                    UTI_thr_now = sum(UTI_thr_now)/len(UTI_thr_now) # average throughput for all users
            else:
                UTI_thr_now = 0   
            
        # save trace as animation here
        # if n%20 == 0: # plot and save png per 200 ms
        #     if n == 0:
        #         count = 1
        #     trace_plot(net, n, count, LB_mode, UTI_update_instances, LB_update_instances) # for debugging
        #     count = count + 1 
        
        UTI_thr_list.append(UTI_thr_now)     
        
    UTI_thr_list = [x for x in UTI_thr_list if x != 0]
        
    UTI_aver_thr = sum(UTI_thr_list)/len(UTI_thr_list)

    results = [UTI_aver_thr, UTI_thr_list]
    return results


# for finding instant throughput with UTI
def start_simualtion_instantUTI(net, UTI, Trained_Net, M, opt_mode):

    Delta_time = int(UTI/0.01)
    
    if UTI == 0.01:
        # start position
        net.mobility_trigger(0, net.X_list_new, net.Y_list_new, net.Saved_SNR_list) # update SNRs and Capacity
        net.load_balancing(Trained_Net, M)
        Rho_iu = PA_optimization(net.B, net.AP_num, net.UE_num, net.X_iu, net.R_requirement, net.Capacity, opt_mode)
        Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num
        # target UE throughput
        for i in range(1):
            list1 = net.Capacity[i]
            list2 = Rho_transposed[i]
            thr_ideal = min(sum(list(np.multiply(list1, list2))), net.R_requirement[i]) #
        
        # save initial connected UE info for the AP that connects target UE
        connected_UE = []
        R_connected_UE = []
        SNR_connected_UE = []
        AP_tar_index = net.X_iu[0] # target AP
        for i in range(net.UE_num):
            if net.X_iu[i] == AP_tar_index:
                connected_UE.append(i)
                R_connected_UE.append(net.R_requirement[i])
                SNR_connected_UE.append(net.SNR_matrix[i][AP_tar_index-1])
                
        SINR_tar = net.SNR_matrix[0]
        SINR_now = SINR_tar[AP_tar_index-1]
        SNR = SINR_now # record the initial SNR value between target UE and AP
        ##### record distance here
        tar_position = net.recorded_trace[0][0]
        AP_position = net.AP_position[AP_tar_index - 1] ## 2-D distance
        d = distance(tar_position, AP_position[0:2]) ## 2-D distance
        ##### record direction here
        tar_position_new = net.recorded_trace[1][0]
        v1 = [(tar_position_new[0]-tar_position[0]), (tar_position_new[1]-tar_position[1])]
        v2 = [(AP_position[0]-tar_position[0]), (AP_position[1]-tar_position[1])]
        theta = angle2(v1, v2)
    else:
        # when after UTI
        net.mobility_trigger(0, net.X_list_new, net.Y_list_new, net.Saved_SNR_list) # update SNRs and Capacity
        net.load_balancing(Trained_Net, M)
        net.mobility_trigger(Delta_time, net.X_list_new, net.Y_list_new, net.Saved_SNR_list) # update SNRs and Capacity
        Rho_iu = PA_optimization(net.B, net.AP_num, net.UE_num, net.X_iu, net.R_requirement, net.Capacity, opt_mode)
        Rho_transposed = np.matrix(Rho_iu).T.tolist() # UE_num * AP_num
        # target UE throughput
        for i in range(1):
            list1 = net.Capacity[i]
            list2 = Rho_transposed[i]
            thr_UTI = min(sum(list(np.multiply(list1, list2))), net.R_requirement[i]) #
    
    if UTI == 0.01:
        result = [thr_ideal, 0, connected_UE, R_connected_UE, SNR_connected_UE, d, theta, SNR]
    else:
        result = [0, thr_UTI, 0, 0, 0, 0, 0, 0]
    return result

# for finding 5% performance gap UTI (instant throughput gap, not average)
def start_simualtion_UTI(net, Trained_Net, M, opt_mode, Ideal_thr, gap_target):
    ###### 
    value1 = Ideal_thr*(1-gap_target[0]/100) # 3% Gap
    value2 = Ideal_thr*(1-gap_target[1]/100) # 5% Gap
    
    UTI_list = [i/100 + 0.02 for i in range(199)]
    UTI_thr_list = []
    state = 1
    state1 = 1
    
    for UTI_delay in UTI_list:
        Delta_time = int(UTI_delay/0.01)
        
        thr_UTI = start_simualtion_instantUTI(net, UTI_delay, Trained_Net, M, opt_mode)
        thr_UTI_now = thr_UTI[1]
        UTI_thr_list.append(thr_UTI_now)
        
        if state1 == 1:
            if thr_UTI_now < value1: # 3% Gap
                UTI_Gap3 = UTI_delay - 0.01
                state1 = 0
                    
        if thr_UTI_now < value2: # 5% Gap
            UTI_Gap5 = UTI_delay - 0.01
            state = 0
            
        if UTI_delay == 2:
            if state1 == 1:
                UTI_Gap3 = 2
            UTI_Gap5 = 2
            state = 0
        
        if state == 0:
            break
        
    print('############ Gap3 is %s, and Gap5 is %s #############' % (UTI_Gap3, UTI_Gap5))   
    
    result = [UTI_thr_list, UTI_Gap3, UTI_Gap5]
    return result

class CsvUTI():
    def __init__(self,
                 ):
        self.title = ['Test','x0(m)','y0(m)','d(m)','SNR(dB)','direction_theta(rad)','R_Tar(Mbps)','V_Tar(m/s)','UE_num','V_Cond','Connected UE_num for target AP','R_targetAP',
                      'SNR_Connected_targetAP', 'Ideal_Thr(Target)', 'UTI_Thr_List(per10ms)','UTI(3%)','UTI(5%)','Runtime']
        self.data = []        
    def update(self, log, path):
        self.data.append(log)
        dataframe = pd.DataFrame(data=self.data, columns=self.title)
        dataframe.to_csv(path, index=False, sep=',')

class CsvLoss(CsvUTI):
    def __init__(self,
                 ):
        self.title = ['Trail', 'Trained Loss', 'Test Loss']
        self.data = []

class CsvPreUTI(CsvUTI):
    def __init__(self,
                 ):
        self.title = ['Trail', 'Type', 'Real_UTI', 'Mixed_Pre_UTI', 'SingleAP_Pre_UTI']
        self.data = []

class CsvGap(CsvUTI):
    def __init__(self,
                 ):
        self.title = ['Trail','Connected_UE','Ideal_Thr','Delayed_Thr','Predicted_UTI', 'Gap', 'Gap(200ms)','Gap(1000ms)','Tar_SNR','theta','R_Tar', 'V_Tar', 'V_Cond','UE_num','R_targetAP','SNR_targetAP', 'Runtime']
        self.data = []

class CsvFig1(CsvUTI):
    def __init__(self,
                 ):
        self.title = ['Trail','UE_num', 'Aver_Velocity(m/s)', 'ATCNN@10ms(Mbps)', 'MS-ATCNN(Mbps)','ATCNN@200ms(Mbps)','ATCNN@AverUTI(Mbps)', 'ATCNN@1000ms(Mbps)','Runtime']
        self.data = []

class CsvFig2(CsvUTI): # without delay
    def __init__(self,
                 ):
        self.title = ['Trail', 'UE_num', 'Aver_Velocity(m/s)', 'MS-ATCNN w/wo delay', 'ATCNN@AverUTI w/wo delay', 'GT@AverUTI w/o delay', 'GT@AverUTI with delay', 'SSS@AverUTI w/wo delay', 'Runtime']
        self.data = []

def UTI_fitting_function(UTI, Ideal_thr_aver, Delay_Thr_aver, gap_target, order):
    # polynomial order
    gap = []
    for i in range(len(Delay_Thr_aver)):
        gap.append((Ideal_thr_aver - Delay_Thr_aver[i])/Ideal_thr_aver*100)
    
    fit_equation = P.fit(UTI, gap, order)
    UTI = (fit_equation - gap_target).roots() 
    real_delay = UTI[abs(UTI.imag)<0.001].real
    true_delay = real_delay[real_delay>0]
    true_delay = true_delay[true_delay<2]
    if len(true_delay) > 1:
        true_delay = [0];
    if len(true_delay) == 0:
        true_delay = [2];
    return true_delay

def mobility_trace_hotspot(UE_num, x_length, y_length, target_velocity, condition_velocity, dis, total_time, target_AP, hotspot_num_list):
    
    hotspot_num = random.choice(hotspot_num_list)
    
    ############ generate trace
    target_trace = []
    condition_trace = []
    time_index = 0
    recorded_trace = []
        
    waypoints_num = int(total_time/0.01 + 1)
    
    dis = 4
    while dis >= 3:
        n = 0
        rw_target = random_walk(1, dimensions=(x_length, y_length), velocity=(target_velocity), distance=dis)
        for positions in rw_target:
            if n == 1: # 101 points for simulating 10 seconds with time interval of 100ms
                break
            else:
                start_point = positions.tolist()
                n += 1
        start_point = start_point[0]
        dis = distance(start_point, target_AP)
        
    cond_trace_list = []
    if UE_num >= 10:
        for i in range(hotspot_num-1):
            dis = 3
            while dis >= 2:
                cond_trace = []
                n = 0
                rw_cond = random_walk(1, dimensions=(x_length, y_length), velocity=(condition_velocity), distance=dis)
                for positions in rw_cond:
                    if n == waypoints_num: # 101 points for simulating 10 seconds with time interval of 100ms
                        break
                    else:
                        cond_trace.append(positions.tolist())
                        n += 1
                start_point = cond_trace[0][0]
                dis = distance(start_point, target_AP)
            cond_trace_list.append(cond_trace)
    
    rw_condition = random_walk(UE_num - 1, dimensions=(x_length, y_length), velocity=(condition_velocity), distance=dis)
    
    for positions in rw_target:
        if time_index == waypoints_num: # 101 points for simulating 10 seconds with time interval of 100ms
            break
        else:
            target_trace.append(positions.tolist())
            time_index += 1
        
    time_index = 0
    for positions in rw_condition:
        if time_index == waypoints_num: # 101 points for simulating 10 seconds with time interval of 100ms
            break
        else:
            condition_trace.append(positions.tolist())
            time_index += 1
    
    if UE_num >= 10:
        # replace condition UE trace
        for i in range(hotspot_num-1):
            for ii in range(waypoints_num):
                condition_trace[ii][-i-1] = cond_trace_list[i][ii][0]

    for j in range(waypoints_num):
        condition_trace[j].append(target_trace[j][0])
        condition_trace[j].reverse()
    
    recorded_trace = condition_trace # first position in time sequence is target UE
    return recorded_trace



def mobility_trace(UE_num, x_length, y_length, Velocity, dis, total_time):
    ############ generate trace
    time_index = 0
    recorded_trace = []
    
    waypoints_num = int(total_time/0.01 + 1)
    
    for i in range(len(Velocity)):
        rw_target = random_walk(1, dimensions=(x_length, y_length), velocity=(Velocity[i]), distance=dis)
        trace = []
        for positions in rw_target:
            if time_index == waypoints_num: # 101 points for simulating 10 seconds with time interval of 100ms
                break
            else:
                trace.append(positions.tolist())
                time_index += 1

        time_index = 0
        if i == 0:
             recorded_trace = trace
        else:
             for j in range(waypoints_num):
                recorded_trace[j].append(trace[j][0])
    
    return recorded_trace


def trace_plot(net, n, fig_num, mode, UTI_update_instances, LB_update_instances):
    x_list = []
    y_list = []
    trace = net.recorded_trace
    AP_position = net.AP_position
    
    for i in range(n+1):    
        x_list.append(trace[i][0][0])
        y_list.append(trace[i][0][1])
    
    # plotting the trace 
    plt.plot(x_list, y_list, linestyle='--', color='b')
    plt.xlim(0, 16)
    plt.ylim(0, 16)
    plt.axvline(x=4, color='k', linestyle='--', linewidth=1)
    plt.axvline(x=8, color='k', linestyle='--', linewidth=1)
    plt.axvline(x=12, color='k', linestyle='--', linewidth=1)
    
    # add AP area lines
    plt.axhline(y=4, color='k', linestyle='--', linewidth=1)
    plt.axhline(y=8, color='k', linestyle='--', linewidth=1)
    plt.axhline(y=12, color='k', linestyle='--', linewidth=1)
    
    # plot all AP points 
    for i in range(len(AP_position)):    
        plt.scatter(AP_position[i][0], AP_position[i][1], color='black', marker='o', s=40)
        if i == 0:
            plt.text(AP_position[i][0], AP_position[i][1], 'WiFi', ha='left', va='top', fontsize = 8)
        else:
            plt.text(AP_position[i][0], AP_position[i][1], 'LiFi', ha='left', va='top', fontsize = 8)
    
    # plot all users and connection lines
    for i in range(net.UE_num):       
        AP_index = net.X_iu[i] - 1
        if i == 0:
            plt.scatter(trace[n][i][0], trace[n][i][1], color='red', marker='*', s=40)
            plt.text(trace[n][i][0], trace[n][i][1], '%s'%(int(net.R_requirement[i])), ha='left', va='bottom', color='r', size=8)
            plt.plot([trace[n][i][0], AP_position[AP_index][0]], [trace[n][i][1], AP_position[AP_index][1]], color='r', linestyle='-', linewidth=2)
        else:
            plt.scatter(trace[n][i][0], trace[n][i][1], color='green', marker='s', s=20)
            plt.text(trace[n][i][0], trace[n][i][1], '%s'%(int(net.R_requirement[i])), ha='left', va='bottom', color='g', size=7)
            plt.plot([trace[n][i][0], AP_position[AP_index][0]], [trace[n][i][1], AP_position[AP_index][1]], color='g', linestyle=':', linewidth=1.5)
    
    # plot LB update frequence and delay time
    for i in range(len(LB_update_instances)):
        time_index = LB_update_instances[i]    
        plt.scatter(x_list[time_index], y_list[time_index], color='m', marker='d', s=18)
    for i in range(len(UTI_update_instances)):
        time_index = UTI_update_instances[i]
        plt.scatter(x_list[time_index], y_list[time_index], color='k', marker='|', s=40)


    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')
    if mode == 'ATCNN':
        plt.title('MS-ATCNN, Target %s m/s, Condition %s m/s, time at %s ms'%(int(net.Velocity_Tar*100), int(net.condition_velocity*100), n*10))
    elif mode == 'SSS':
        plt.title('SSS, Target %s m/s, Condition %s m/s, time at %s ms'%(int(net.Velocity_Tar*100), int(net.condition_velocity*100), n*10))
    else:
        plt.title('GT, Target %s m/s, Condition %s m/s, time at %s ms'%(int(net.Velocity_Tar*100), int(net.condition_velocity*100), n*10))
        
    
    if mode == 'ATCNN':
        plt.savefig("./Plots/Trace_gif_ATCNN/frame%s.png"%fig_num, dpi=300)
    elif mode == 'SSS':
        plt.savefig("./Plots/Trace_gif_SSS/frame%s.png"%fig_num, dpi=300)  
    else:
        plt.savefig("./Plots/Trace_gif_GT/frame%s.png"%fig_num, dpi=300)
    # plt.show()  
    plt.close()          

def gif_plt(png_num, LB_mode, trail_time):
    images = []
    for k in range(png_num):
        filename = './Plots/Trace_gif_%s/frame%s.png'%(LB_mode, k+1) 
        images.append(imageio.imread(filename))
    imageio.mimwrite('./Plots/Trace_gif_%s/%s_trace%s.gif'%(LB_mode, LB_mode, trail_time), images, duration = 200, loop = 1)
    

def call_function(net, time_index, X_iu_old, UE_index, mode=None):
    # Type 1
    max_tar_SNR_AP1 = 20.90560832
    min_tar_SNR_AP1 = 4.433023854
    # Type2
    max_tar_SNR_AP2 = 23.01966378
    min_tar_SNR_AP2 = 5.34853973
    # Type3
    max_tar_SNR_AP3 = 23.21163127
    min_tar_SNR_AP3 = 10.7606035
    # Type4
    max_tar_SNR_AP4 = 65.3282696916349
    min_tar_SNR_AP4 = -0.871427041
    
    AP_tar = X_iu_old[UE_index]
    
    if mode == None:
        if AP_tar == 1:
            min_tar_SNR = min_tar_SNR_AP4
            max_tar_SNR = max_tar_SNR_AP4
            loaded_model = net.loaded_model_Type4
        elif AP_tar in [7, 8, 11, 12]:
            min_tar_SNR = min_tar_SNR_AP1
            max_tar_SNR = max_tar_SNR_AP1
            loaded_model = net.loaded_model_Type1
        elif AP_tar in [2, 5, 14, 17]:
            min_tar_SNR = min_tar_SNR_AP3
            max_tar_SNR = max_tar_SNR_AP3
            loaded_model = net.loaded_model_Type3
        else:
            min_tar_SNR = min_tar_SNR_AP2
            max_tar_SNR = max_tar_SNR_AP2
            loaded_model = net.loaded_model_Type2
    else:
        loaded_model = net.loaded_model_Mixed
        max_tar_SNR =  65.32826969
        min_tar_SNR = 1.435117189
    
    SINR_tar = net.SNR_matrix[UE_index]
    SNR = SINR_tar[AP_tar-1]
    tar_position = net.recorded_trace[time_index][UE_index]
    tar_position_new = net.recorded_trace[time_index+1][UE_index]
    AP_position = net.AP_position[AP_tar - 1] ## 2-D distance
    v1 = [(tar_position_new[0]-tar_position[0]), (tar_position_new[1]-tar_position[1])]
    v2 = [(AP_position[0]-tar_position[0]), (AP_position[1]-tar_position[1])]
    theta = angle2(v1, v2)
    R_tar = net.R_requirement[UE_index]
    Velocity_UE = net.Velocity[UE_index]

    input_now = [[(SNR-min_tar_SNR)/(max_tar_SNR-min_tar_SNR), theta/math.pi, R_tar/500, Velocity_UE/0.1]] # normalization 

    input_now = np.array(input_now)
    pre_output = loaded_model.predict(input_now).tolist()
    predicted_UTI = pre_output[0][0]*2 # normalization
    predicted_UTI = np.clip([predicted_UTI], 0.01, 2).tolist()

    return predicted_UTI[0]