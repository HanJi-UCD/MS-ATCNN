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

############ load trained ATCNN model, 9 LiFi AP case ############
Trained_Net = ATCNN(input_dim=(AP_num+1), cond_dim=M*(AP_num+1), cond_out_dim=(AP_num+1), output_dim=AP_num)
Trained_Net.load_state_dict(torch.load("./trained_model/ATCNN_16LiFi_10m_100UE.pth",map_location=torch.device('cpu')), strict=False) # 1000 batch case  
device = torch.device('cpu')
Trained_Net.eval()
Trained_Net.to(device)
print('************************ Finished Loading Trained ATCNN *****************************')
tf.keras.utils.disable_interactive_logging()
net.loaded_model_Mixed = tf.keras.models.load_model('trained_model/NN_UTI_100UE_MixedType.h5')

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
    UTI = call_function(net, 0, X_iu_old, 0, 'Mixed')
    # log.append([net.Velocity[0], UTI])
    
    if i%10 == 0:
        print('i = ', i)
        
    if AP_tar in (7, 8, 11, 12): # Type I
        log = [i, 'I', UTI]
        csv1.update(log, 'result/Mixed_PreUTI_Type1.csv') # save results
    elif AP_tar in (3, 4, 6, 9, 10, 13, 15, 16): # Type II
        log = [i, 'II', UTI]
        csv2.update(log, 'result/Mixed_PreUTI_Type2.csv') # save results
    elif AP_tar in (2, 5, 14, 17): # Type III
        log = [i, 'III', UTI]
        csv3.update(log, 'result/Mixed_PreUTI_Type3.csv') # save results
    else: # Type IV
        log = [i, 'IV', UTI]    
        csv4.update(log, 'result/Mixed_PreUTI_Type4.csv') # save results


#%%
# get UTI histgram data
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import math
import csv

############### read raw csv file and pre-process data
file_name = 'dataset/Optimal_UTI4/16AP_dataset_UTI_100UE_new_Type4.xlsx'
# choose sample number here
# df = pd.read_excel(file_name, nrows=500, dtype={'R_targetAP': 'object', 'SNR_Connected_targetAP': 'object'})
df = pd.read_excel(file_name, dtype={'R_targetAP': 'object', 'SNR_Connected_targetAP': 'object'})
df = df[df['V_Tar(m/s)']!=0] # remove 0m/s samples as theta is None 

shuffled_df = df.sample(frac=1) 

def flatten_list(my_list):
    new_list = []
    for item in my_list:
        if isinstance(item, list):
            new_list.extend(flatten_list(item))
        else:
            new_list.append(item)
    return new_list

def calc_interval_prob(lst, interval):
    n = len(lst)
    k = sum(1 for x in lst if interval[0] <= x <= interval[1])
    prob = k / n
    return prob

loaded_model = tf.keras.models.load_model('trained_model/NN_UTI_100UE_new_Type4.h5')

# revise normalization parameters accordingly here
# Type 1
# max_tar_SNR = 20.90560832
# min_tar_SNR = 4.433023854
# df = shuffled_df[0:1000]
# Type2
# max_tar_SNR = 23.01966378
# min_tar_SNR = 5.34853973
# df = shuffled_df[0:1000]
# Type3
# max_tar_SNR = 23.21163127
# min_tar_SNR = 10.7606035
# df = shuffled_df[0:1000]
# Type4
max_tar_SNR = 65.3282696916349
min_tar_SNR = -0.871427041
df = shuffled_df[0:1000]

print('Length of unseen test dataset is:', len(df))

# Target inputs
Tar_inputs = df.iloc[:, [4, 5, 6, 7]].values.tolist()
# Delay outputs
real_outputs = df.iloc[:, [16]].values.tolist()

# normalize target inputs
Tar_inputs = np.array(Tar_inputs)
Tar_inputs[:, 0] = (Tar_inputs[:, 0] - min_tar_SNR)/(max_tar_SNR - min_tar_SNR)
Tar_inputs[:, 1] = Tar_inputs[:, 1]/math.pi
Tar_inputs[:, 2] = Tar_inputs[:, 2]/500
Tar_inputs[:, 3] = Tar_inputs[:, 3]/10
Tar_inputs = Tar_inputs.tolist()

new_inputs = [x for x in zip(Tar_inputs)]

new_flatten_inputs = []
for i in range(len(new_inputs)):
    new_flatten_inputs.append(flatten_list(new_inputs[i]))

final_input = np.array(new_flatten_inputs)

pre_output = loaded_model.predict(final_input).tolist()

pre_output_list = []
for i in range(len(pre_output)):
    pre_output_list.append(pre_output[i][0]*2)

print('Predicted Output:', pre_output_list)
print('Real output should be:', real_outputs)

x = list(range(0, len(real_outputs)))
fig1 = plt.figure()
plt.scatter(x, pre_output_list, marker='x', color='red', label='Predicted Delay')
plt.scatter(x, real_outputs, marker='o', color='blue', label='True Delay')
plt.legend()
plt.grid(True)
plt.xticks(range(min(x), max(x)+1, 5))
plt.xlabel('Sample index')
plt.ylabel('5% Gap Delay (s)')

# show errors figure
fig2 = plt.figure()
MS_errors = [x[0] - y for x, y in zip(real_outputs, pre_output_list)]

############# Mixed model for predicting UTI
loaded_mixed_model = tf.keras.models.load_model('trained_model/NN_UTI_100UE_MixedType.h5')
max_tar_SNR =  65.32826969
min_tar_SNR = 1.435117189

pre_output = loaded_mixed_model.predict(final_input).tolist()

pre_output_list = []
for i in range(len(pre_output)):
    pre_output_list.append(pre_output[i][0]*2)

print('Predicted Output:', pre_output_list)
print('Real output should be:', real_outputs)

x = list(range(0, len(real_outputs)))
fig1 = plt.figure()
plt.scatter(x, pre_output_list, marker='x', color='red', label='Predicted Delay')
plt.scatter(x, real_outputs, marker='o', color='blue', label='True Delay')
plt.legend()
plt.grid(True)
plt.xticks(range(min(x), max(x)+1, 5))
plt.xlabel('Sample index')
plt.ylabel('5% Gap Delay (s)')

# show errors figure
fig3 = plt.figure()
Mixed_errors = [x[0] - y for x, y in zip(real_outputs, pre_output_list)]

with open('result/Histgram_UTI_Errors_Type4.csv', 'w', newline='') as file:
    # Step 4: Using csv.writer to write the list to the CSV file
    writer = csv.writer(file)
    writer.writerow(MS_errors) # Use writerow for single list
    writer.writerow(Mixed_errors)
