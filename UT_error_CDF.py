# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 20:12:35 2023

@author: 14488
"""

import json
import numpy as np
import torch
from ATCNN_model import ATCNN_16LiFi_100UE as ATCNN
from mytopo import Net, start_simualtion_UTI, start_simualtion_instantUTI, CsvPreUTI, mobility_trace, call_function
import os
import time
import matlab.engine
import tensorflow as tf
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

############# input values ############
UE_num = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
Aver_Velovity = [3]
R_aver = 100 # fix R as 100 M with Gamma distribution

AP_num = 17 # fixed value
X_length = 10
Y_length = 10
dis = 20
opt_mode = 0 # PA mode (1: Optimization method; 0: Approximation method using 1/N)
M = 100

Trail_times = 500
total_time = 10 # seconds

############ Env Setup ############
net = Net(AP_num, 5)

csv = CsvPreUTI()

print('Activating the Matlab engine')
eng = matlab.engine.start_matlab()
print('Successfully activated the Matlab engine')

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
    
############ load trained ATCNN model, 9 LiFi AP case ############
Trained_Net = ATCNN(input_dim=(AP_num+1), cond_dim=M*(AP_num+1), cond_out_dim=(AP_num+1), output_dim=AP_num)
Trained_Net.load_state_dict(torch.load("./trained_model/ATCNN_16LiFi_10m_100UE.pth",map_location=torch.device('cpu')), strict=False) # 1000 batch case  
device = torch.device('cpu')
Trained_Net.eval()
Trained_Net.to(device)
print('************************ Finished Loading Trained ATCNN *****************************')
################################# MAIN ########################################
index = 0

# load UTI models
tf.keras.utils.disable_interactive_logging()
net.loaded_model_Mixed = tf.keras.models.load_model('trained_model/NN_UTI_100UE_MixedType.h5')

net.loaded_model_Type4 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type4.h5')
net.loaded_model_Type1 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type1.h5')
net.loaded_model_Type3 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type3.h5')
net.loaded_model_Type2 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type2.h5')

for i in range(Trail_times):
    ############ save data into log ############
    index += 1
    print('')
    print('******************** ###### For Test %s ###### ********************'%(index))
    # for mobility simulation
    net.waypoints_num = int(total_time/0.01 + 1)
    ############ generate different target velocity ############
    UE_num_now = np.random.choice(UE_num, size=1).tolist() 
    net.UE_num = UE_num_now[0]
    ###### set velocity
    net.Aver_Velocity = np.random.choice(Aver_Velovity).tolist() # randomly choose average velocity
    if net.Aver_Velocity == 1:
        Velocity = np.linspace(0.1/100, 2.0/100, num=20).tolist() # Aver_Velovity = 1
    elif net.Aver_Velocity == 1.5:
        Velocity = np.linspace(0.1/100, 3.0/100, num=30).tolist() # Aver_Velovity = 1.5
    elif net.Aver_Velocity == 2:   
        Velocity = np.linspace(0.1/100, 4.0/100, num=40).tolist() # Aver_Velovity = 2
    elif net.Aver_Velocity == 2.5:
        Velocity = np.linspace(0.1/100, 5.0/100, num=50).tolist() # Aver_Velovity = 2.5
    elif net.Aver_Velocity == 3:
        Velocity = np.linspace(0.1/100, 6.0/100, num=60).tolist() # Aver_Velovity = 3
    elif net.Aver_Velocity == 4:
        Velocity = np.linspace(0.1/100, 8.0/100, num=80).tolist() # Aver_Velovity = 4
    else:
        Velocity = np.linspace(0.1/100, 10.0/100, num=100).tolist() # Aver_Velovity = 5
    net.Velocity = np.random.choice(Velocity, size=net.UE_num).tolist() # randomly choose velocity    
    ############ generate data rate requirement in each trail ############
    R_raw = np.random.gamma(1, R_aver, net.UE_num)
    net.R_requirement = np.clip(R_raw, 10, 500).tolist()

    ############ update trace
    net.recorded_trace = mobility_trace(net.UE_num, X_length, Y_length, net.Velocity, dis, total_time)  
    ############ ideal UTI with 10ms ###################
    tic = time.time()
    
    results_ideal = start_simualtion_instantUTI(net, 0.01, Trained_Net, M, opt_mode) # assume runtime for training is 0   
    Ideal_thr = results_ideal[0]
    ############# Find 5% Gap UTI ######################
    results_UTI = start_simualtion_UTI(net, Trained_Net, M, opt_mode, Ideal_thr, [3,5]) # assume runtime for training is 0 

    Real_UTI = results_UTI[2]
    
    # mobility model
    net.mobility_trigger(0, net.X_list_new, net.Y_list_new, net.Saved_SNR_list) # update SNRs and Capacity
    net.load_balancing(Trained_Net, M)
    X_iu_old = net.X_iu # initial result, first update at t0
    AP_tar = net.X_iu[0]
    MixedAP_UTI = call_function(net, 0, X_iu_old, 0, 'Mixed')
    
    SingleAP_UTI = call_function(net, 0, X_iu_old, 0)
        
    if AP_tar in (7, 8, 11, 12): # Type I
        log = [i, 'I', Real_UTI, MixedAP_UTI, SingleAP_UTI]
    elif AP_tar in (3, 4, 6, 9, 10, 13, 15, 16): # Type II
        log = [i, 'II', Real_UTI, MixedAP_UTI, SingleAP_UTI]
    elif AP_tar in (2, 5, 14, 17): # Type III
        log = [i, 'III', Real_UTI, MixedAP_UTI, SingleAP_UTI]
    else: # Type IV
        log = [i, 'IV', Real_UTI, MixedAP_UTI, SingleAP_UTI]   
    
    csv.update(log, 'result/Fig6_UTI_error_CDF1.csv') # save results
   
    toc = time.time()
                    
    print('For one trail the runtime is:',(toc-tic))