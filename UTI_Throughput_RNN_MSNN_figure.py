#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 18:48:13 2024

@author: han
"""

import json
import numpy as np
import torch
from ATCNN_model import ATCNN_16LiFi_100UE as ATCNN, MSNN, RNN
from mytopo import Net, start_simualtion, CsvFig3, mobility_trace
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

############# input values ############
UE_num = [20, 40, 60, 80, 100]
Aver_Velovity = [2]
R_aver = 100 # fix R as 100 M with Gamma distribution

AP_num = 17 # fixed value
X_length = 10
Y_length = 10
opt_mode = 0 # PA mode (1: Optimization method; 0: Approximation method using 1/N)
M = 100

mobility_mode = 'RandomWalk'

Trail_times = 100
total_time = 5 # seconds

############ Env Setup ############
net = Net(AP_num, 5)
csv = CsvFig3()

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

# tf.keras.utils.disable_interactive_logging()
# net.loaded_model_Type4 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type4.h5')
# net.loaded_model_Type1 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type1.h5')
# net.loaded_model_Type3 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type3.h5')
# net.loaded_model_Type2 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type2.h5')
    
# load trained UTI models in advance
# saved_mode_path1 = './result/temporal_UTI/2024-07-04-06-19-25-PM-MSNN-Type1/final_model.pth'
# saved_mode_path2 = './result/temporal_UTI/2024-07-04-06-21-34-PM-MSNN-Type2/final_model.pth'
# saved_mode_path3 = './result/temporal_UTI/2024-07-04-06-22-58-PM-MSNN-Type3/final_model.pth'
# saved_mode_path4 = './result/temporal_UTI/2024-07-04-06-23-56-PM-MSNN-Type4/final_model.pth'
    
saved_mode_path1 = './result/2024-07-04-08-57-59-PM-MSNN-Type1/final_model.pth'
saved_mode_path2 = './result/2024-07-04-09-00-24-PM-MSNN-Type2/final_model.pth'
saved_mode_path3 = './result/2024-07-04-09-01-26-PM-MSNN-Type3/final_model.pth'
saved_mode_path4 = './result/2024-07-04-09-02-00-PM-MSNN-Type4/final_model.pth'
    
device = torch.device('cpu')
net.loaded_model_Type1 = MSNN(input_dim=3, output_dim=1)
net.loaded_model_Type1.load_state_dict(torch.load(saved_mode_path1, map_location=device), strict=False)
net.loaded_model_Type1.eval()
net.loaded_model_Type1.to(device)
net.loaded_model_Type2 = MSNN(input_dim=3, output_dim=1)
net.loaded_model_Type2.load_state_dict(torch.load(saved_mode_path2, map_location=device), strict=False)
net.loaded_model_Type2.eval()
net.loaded_model_Type2.to(device)
net.loaded_model_Type3 = MSNN(input_dim=3, output_dim=1)
net.loaded_model_Type3.load_state_dict(torch.load(saved_mode_path3, map_location=device), strict=False)
net.loaded_model_Type3.eval()
net.loaded_model_Type3.to(device)
net.loaded_model_Type4 = MSNN(input_dim=3, output_dim=1)
net.loaded_model_Type4.load_state_dict(torch.load(saved_mode_path4, map_location=device), strict=False)
net.loaded_model_Type4.eval()
net.loaded_model_Type4.to(device)
print('************************ Finished Loading Trained MSNN *****************************')

RNN_model_path = './result/temporal_UTI/2024-07-04-06-12-33-PM-RNN/final_model.pth'
device = torch.device('cpu')
net.loaded_model_RNN= RNN(3, 32, 1, 3, 0.5)
net.loaded_model_RNN.load_state_dict(torch.load(RNN_model_path, map_location=device), strict=False)   
net.loaded_model_RNN.eval()
net.loaded_model_RNN.to(device)
print('************************ Finished Loading Trained RNN *****************************')

################################# MAIN ########################################
index = 0

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
    net.recorded_trace = mobility_trace(net.UE_num, X_length, Y_length, net.Velocity, total_time, mobility_mode)  
    ############ ideal UTI with 10ms ###################
    tic = time.time()
    
    results_MSNN = start_simualtion(net, Trained_Net, M, opt_mode, 'Dynamic', 'ATCNN', 'Average')
    print('For MS-ATCNN UTI, average throughput is:', results_MSNN[0])
    
    results_RNN = start_simualtion(net, Trained_Net, M, opt_mode, 'Dynamic', 'RNN', 'Average', RNN_mode='True')
    print('For RNN-ATCNN UTI, average throughput is:', results_RNN[0])
    
    toc = time.time()
    
    log = [i, net.UE_num, net.Aver_Velocity, results_MSNN[0], results_RNN[0], toc-tic]
    
    csv.update(log, 'result/Fig2_Thr_RNN_MSNN_RandomWalk.csv')
                    
    print('For one trail the runtime is:',(toc-tic))