# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 20:29:49 2024

@author: Han
"""

import json
import numpy as np
import torch
from ATCNN_model import ATCNN_16LiFi_100UE as ATCNN
from mytopo import Net, start_simualtion_instantUTI, start_simualtion_UTI, CsvUTI, mobility_trace
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

############# input values ############
Velocity = (1/100, 10/100) # unit of m/s: 1~10 m/s
UE_num = [50] # a fixed UE number for this experiment
R_condition = 50 # fix R as 100 M with Gamma distribution
gap_target = [3, 5] # unit of percentage
R_tar = 50

AP_num = 17 # fixed value
X_length = 10
Y_length = 10

opt_mode = 0 # PA mode (1: Optimization method; 0: Approximation method using 1/N)
M = 100 # ATCNN with max 100 UEs

total_time = 1011 # seconds
sampling_time = 0.1 # sample time series data for every 100 ms 
Trail_times = 10100
mobility_mode = 'GaussMarkov'

############ Env Setup ############
net = Net(AP_num, 5)
csv1 = CsvUTI()
csv2 = CsvUTI()
csv3 = CsvUTI()
csv4 = CsvUTI()

############ import saved SNR values ############
X_list = np.linspace(0, net.X_length, num=(int(net.X_length/0.1) + 1))
Y_list = np.linspace(0, net.Y_length, num=(int(net.Y_length/0.1) + 1))
net.X_list_new = []
for i in range(len(X_list)-1):
    net.X_list_new.append((X_list[i] + X_list[i+1])/2)
net.Y_list_new = []
for i in range(len(Y_list)-1):
    net.Y_list_new.append((Y_list[i] + Y_list[i+1])/2)    
with open('./Saved_SINR_Library/Saved_SINR_info_16AP_10m.json', 'rb') as fp:
    net.Saved_SNR_list = json.load(fp)
    
############ load trained ATCNN model, 16 LiFi AP case ############
Trained_Net = ATCNN(input_dim=(AP_num+1), cond_dim=M*(AP_num+1), cond_out_dim=(AP_num+1), output_dim=AP_num)
Trained_Net.load_state_dict(torch.load("./trained_model/ATCNN_16LiFi_10m_100UE.pth",map_location=torch.device('cpu')), strict=False) # 1000 batch case  
device = torch.device('cpu')
Trained_Net.eval()
Trained_Net.to(device)
print('************************ Finished Loading Trained ATCNN *****************************')
############ Save Simulation Parameters ############
exp_folder1 = os.path.join('./result', "Optimal_UTI1_%sUE_%sM"%(UE_num[0], R_tar)) # revise test1 name every experiment
if not os.path.exists(exp_folder1):
    os.mkdir(exp_folder1)
exp_folder2 = os.path.join('./result', "Optimal_UTI2_%sUE_%sM"%(UE_num[0], R_tar)) # revise test1 name every experiment
if not os.path.exists(exp_folder2):
    os.mkdir(exp_folder2)
exp_folder3 = os.path.join('./result', "Optimal_UTI3_%sUE_%sM"%(UE_num[0], R_tar)) # revise test1 name every experiment
if not os.path.exists(exp_folder3):
    os.mkdir(exp_folder3)
exp_folder4 = os.path.join('./result', "Optimal_UTI4_%sUE_%sM"%(UE_num[0], R_tar)) # revise test1 name every experiment
if not os.path.exists(exp_folder4):
    os.mkdir(exp_folder4)
################################# MAIN ########################################
index = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0

# for mobility simulation
net.waypoints_num = int(total_time/0.01 + 1)
UE_num_now = np.random.choice(UE_num, size=1).tolist() 
net.UE_num = UE_num_now[0]
############ generate different target velocity ############
net.Velocity = Velocity 
R_raw = np.random.gamma(1, R_condition, net.UE_num)
R_raw[0] = R_tar # need R for target UE here, otherwise for UE with small R always connect WiFi
############ generate data rate requirement in each trail ############
net.R_requirement = np.clip(R_raw, 10, 500).tolist()
############ update trace
net.recorded_trace = mobility_trace(net.UE_num, X_length, Y_length, net.Velocity, total_time, mobility_mode)
# mobility_plot(net)
for i in range(Trail_times):
    ############ save data into log ############
    index += 1
    print('')
    print('******************** ###### For Sampling %s data ###### ********************'%(index))
    
    time_slot = i*int(sampling_time/0.01)
    
    tic = time.time()
    net.mobility_trigger(time_slot, net.X_list_new, net.Y_list_new, net.Saved_SNR_list) # update SNRs and Capacity
    net.load_balancing(Trained_Net, M) ############### Use 100UE ATCNN
    AP_tar = net.X_iu[0]
            
    location = net.recorded_trace[time_slot][0] # target location    
    x0 = location[0]
    y0 = location[1]
            
    ############ ideal instant throughput ###################
    results_ideal = start_simualtion_instantUTI(net, 0.01, time_slot, Trained_Net, M, opt_mode) # assume runtime for training is 0   
    Ideal_thr = results_ideal[0]
    d = results_ideal[5]
    theta = results_ideal[6]
    speed = results_ideal[-1]
    ############# Find 5% Gap UTI ######################
    results_UTI = start_simualtion_UTI(net, Trained_Net, M, opt_mode, Ideal_thr, gap_target, time_slot) # assume runtime for training is 0 
        
    UTI_Thr_List = results_UTI[0]
        
    log = [index, x0, y0, d, results_ideal[7], theta, net.R_requirement[0], speed, net.UE_num,  net.Velocity[1:]]  
    log.append(len(results_ideal[2])) #          
    log.append(results_ideal[3]) # record R values                 
    log.append(results_ideal[4]) # record SNR_connected_tarAP                 
    log.append(Ideal_thr)
    log.append(UTI_Thr_List) # delayed thr list            
    log.append(results_UTI[1]) # 3% Gap           
    log.append(results_UTI[2]) # 5% Gap           
    toc = time.time()
    log.append(toc-tic)
        
    print('For one trail the runtime is:',(toc-tic))
        
    if AP_tar in (7, 8, 11, 12): # Type I
        csv1.update(log, f'{exp_folder1}/16AP_log_UTI_100UE_Markov_Type1.csv') # save results
        count1 = count1 + 1
        print('Collected Type I Sample:', count1)
    elif AP_tar in (3, 4, 6, 9, 10, 13, 15, 16): # Type II
        csv2.update(log, f'{exp_folder2}/16AP_log_UTI_100UE_Markov_Type2.csv') # save results
        count2 = count2 + 1
        print('Collected Type II Sample:', count2)
    elif AP_tar in (2, 5, 14, 17): # Type III
        csv3.update(log, f'{exp_folder3}/16AP_log_UTI_100UE_Markov_Type3.csv') # save results
        count3 = count3 + 1
        print('Collected Type III Sample:', count3)
    else: # Type IV
        csv4.update(log, f'{exp_folder4}/16AP_log_UTI_100UE_Markov_Type4.csv') # save results
        count4 = count4 + 1
        print('Collected Type IV Sample:', count4)