# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 18:18:57 2023

@author: Han
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 02:28:06 2022

@author: han
"""
from mytopo import call_function
from ATCNN_model import ATCNN_16LiFi_100UE as ATCNN
import torch
import numpy as np
from mytopo import Net, mobility_trace, CsvPreUTI
import json
import tensorflow as tf

UE_num = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# Velocity = np.linspace(0.1/100, 1.9/100, num=19).tolist() # Aver_Velovity = 1
# Velocity = np.linspace(0.1/100, 2.9/100, num=29).tolist() # Aver_Velovity = 1.5
# Velocity = np.linspace(0.1/100, 4.0/100, num=40).tolist() # Aver_Velovity = 2
# Velocity = np.linspace(0.1/100, 5.0/100, num=50).tolist() # Aver_Velovity = 2.5
# Velocity = np.linspace(0.1/100, 5.9/100, num=59).tolist()   # Aver_Velovity = 3
# Velocity = np.linspace(0.1/100, 8.0/100, num=80).tolist() # Aver_Velovity = 4
Velocity = np.linspace(0.1/100, 10.0/100, num=100).tolist() # Aver_Velovity = 5

R_aver = 100 # fix R as 100 M with Gamma distribution
AP_num = 17 # fixed value
X_length = 10
Y_length = 10
dis = 20
opt_mode = 0 # PA mode (1: Optimization method; 0: Approximation method using 1/N)
M = 100

Trail_times = 2000
total_time = 10 # seconds
net = Net(AP_num, 5)

csv1 = CsvPreUTI()
csv2 = CsvPreUTI()
csv3 = CsvPreUTI()
csv4 = CsvPreUTI()

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
tf.keras.utils.disable_interactive_logging()
net.loaded_model_Type4 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type4.h5')
net.loaded_model_Type1 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type1.h5')
net.loaded_model_Type3 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type3.h5')
net.loaded_model_Type2 = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type2.h5')

# log = []
for i in range(Trail_times):
    UE_num_now = np.random.choice(UE_num, size=1).tolist() 
    net.UE_num = UE_num_now[0]
    net.Velocity = np.random.choice(Velocity, size=net.UE_num).tolist() # randomly choose velocity
    ############ generate data rate requirement in each trail ############
    R_raw = np.random.gamma(1, R_aver, net.UE_num)
    net.R_requirement = np.clip(R_raw, 10, 500).tolist()
    
    ############ update trace
    net.recorded_trace = mobility_trace(net.UE_num, X_length, Y_length, net.Velocity, dis, total_time)  
    # mobility model
    net.mobility_trigger(0, net.X_list_new, net.Y_list_new, net.Saved_SNR_list) # update SNRs and Capacity
    net.load_balancing(Trained_Net, M)
    X_iu_old = net.X_iu # initial result, first update at t0
    AP_tar = net.X_iu[0]
    UTI = call_function(net, 0, X_iu_old, 0)
    # log.append([net.Velocity[0], UTI])
    
    if i%10 == 0:
        print('i = ', i)
        
    if AP_tar in (7, 8, 11, 12): # Type I
        log = [i, 'I', UTI]
        csv1.update(log, 'result/MS_PreUTI_Type1.csv') # save results
    elif AP_tar in (3, 4, 6, 9, 10, 13, 15, 16): # Type II
        log = [i, 'II', UTI]
        csv2.update(log, 'result/MS_PreUTI_Type2.csv') # save results
    elif AP_tar in (2, 5, 14, 17): # Type III
        log = [i, 'III', UTI]
        csv3.update(log, 'result/MS_PreUTI_Type3.csv') # save results
    else: # Type IV
        log = [i, 'IV', UTI]    
        csv4.update(log, 'result/MS_PreUTI_Type4.csv') # save results
    

# log = np.array(log)
# result = np.mean(log, axis=0)
# print('Average velovity is %s, and average UTI is %s'%(result[0], result[1]))

# Average velovity is 1m/s(0-2), and average UTI is 1.0146900708973408
# Average velovity is 1.5m/s(0-3), and average UTI is 0.827367932225267
# Average velovity is 2m/s(0-4), and average UTI is 0.7415552637539804
# Average velovity is 2.5m/s(0-5), and average UTI is 0.6384603425115347
# Average velovity is 3m/s(0-6), and average UTI is 0.6263300363719464
# Average velovity is 4m/s(0-8), and average UTI is 0.489643948180601
# Average velovity is 5m/s(0-10), and average UTI is 0.4427249391237274
# Fixed velovity is 5m/s, and average UTI is 0.3104300149064511
 
 