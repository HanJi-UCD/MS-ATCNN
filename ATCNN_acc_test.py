# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 21:37:25 2022

@author: Han
"""

import torch
import numpy as np
import pandas as pd
import warnings
from utils import mapping, normalization
from ATCNN_model import switch, to_binary, ATCNN_16LiFi_100UE, ATCNN_16LiFi_100UE
warnings.filterwarnings("ignore")

# test normalised and mirrored dataset
def test_acc_ATCNN(input_path, output_path, AP_num, UE_num):
    #print('******Loading Input data******')
    data = pd.read_csv(input_path, header=None)
    values = np.array(data).T.tolist() # with dimension of 256*300
    trained_output = []
    test_UE_num = 1 # revise here
    sample_number = 256

    for i in range(sample_number):
        condition = values[i]
        condition = torch.tensor(condition)
        output_instance = []
        for j in range(test_UE_num):
            raw_condition = torch.tensor(values[i])
            condition_now = switch(raw_condition, j, AP_num+1) # switch j-th UE into last position
            condition_now = torch.tensor([condition_now])
            
            Target = condition_now[..., 0+j*user_dim:(j+1)*user_dim]
            Condition = condition_now[..., 0:M*user_dim] 
            
            # Condition = condition_now[..., 0:]
            
            output = Trained_Net.forward(Target, Condition) 
            binary_output = to_binary(AP_num, output.tolist())
            output_instance.extend(binary_output)
        trained_output.append(output_instance) ###

    #print('******Loading Output data******')
    label = pd.read_csv(output_path, header=None)
    real_output = pd.DataFrame(label)
    real_output = np.array(real_output).T.tolist()
    
    count = 0
    for i in range(sample_number):
        for j in range(test_UE_num):
            real_output_now = real_output[i][0+j*AP_num:(j+1)*AP_num]
            # real_output_now = real_output[i][0+(j+9)*AP_num:(j+10)*AP_num]
            trained_output_now = np.array(trained_output[i][0+j*AP_num:(j+1)*AP_num])
            if all(real_output_now == trained_output_now):  
                pass
            else:
                count = count + 1
    acc = 1 - count/sample_number/test_UE_num
    return acc

# test raw dataset without normalization and mapping
def test_acc_ATCNN_raw(M, input_path, output_path, AP_num, UE_num, test_UE_num, SNR_min):
    #print('******Loading Input data******')
    data = pd.read_csv(input_path, header=None)
    values = np.array(data).T.tolist() # with dimension of 256*300
    trained_output = []
    
    for i in range(256):
        condition = values[i]
        condition = torch.tensor(condition)
        output_instance = []
        for j in range(test_UE_num):
            raw_condition = torch.tensor(values[i])
            condition_now = switch(raw_condition, j, AP_num+1) 
            condition_now = torch.tensor([condition_now])
            
            mirroring_condition = mapping(M-1, [UE_num]*256, condition_now.tolist(), AP_num)
            
            nor_mirroring_condition = normalization(M, AP_num, mirroring_condition, 60, SNR_min, 1000) # nomalization is correct
    
            nor_mirroring_condition = torch.tensor(nor_mirroring_condition).to(torch.float32)
            
            Target = nor_mirroring_condition[..., 0:user_dim]    
            Condition = nor_mirroring_condition[..., 0:]
            
            output = Trained_Net.forward(Target, Condition) 
            binary_output = to_binary(AP_num, output.tolist())
            output_instance.extend(binary_output)
        trained_output.append(output_instance) ###

    #print('******Loading Output data******')
    label = pd.read_csv(output_path, header=None)
    real_output = pd.DataFrame(label)
    real_output = np.array(real_output).T.tolist()

    count = 0
    for i in range(256):
        for j in range(test_UE_num):
            real_output_now = real_output[i][0+j*AP_num:(j+1)*AP_num]
            trained_output_now = np.array(trained_output[i][0+j*AP_num:(j+1)*AP_num])
            if all(real_output_now == trained_output_now):
                pass
            else:
                count = count + 1
    acc = 1 - count/256/test_UE_num         
    return acc

# 16 LiFi size
user_dim = 18 # 
output_dim = 17 # 
cond_dim = 1800 #
SNR_min = -20
M = 100

# 9 LiFi size
# user_dim = 11 # 
# output_dim = 10 # 
# cond_dim = 550 #
# SNR_min = -20

# 4 LiFi size
# user_dim = 6 # 
# output_dim = 5 # 
# cond_dim = 300 #
# SNR_min = 15 # 4 LiFi, 3W case

# Trained_Net = ATCNN(input_dim=user_dim, cond_dim=cond_dim, cond_out_dim=user_dim, output_dim=output_dim)
Trained_Net = ATCNN_16LiFi_100UE(input_dim=user_dim, cond_dim=cond_dim, cond_out_dim=user_dim, output_dim=output_dim)
device = torch.device('cpu')

model_name = "trained_model/TCNN_16LiFi_10m_100UE.pth"

print('******Loading Net******')
print("Testing ATCNN model name is:", model_name)
Trained_Net.load_state_dict(torch.load(model_name,map_location=torch.device('cpu')), strict=False) # 1000 batch case 

# confirmed that the saved net parameters are loaded and unchangable
Trained_Net.eval()
Trained_Net.to(device)
print('******Starting testing different UE number******')

# new test dataset 
# input_path = 'dataset/TCNN_16LiFi_10m_100UE/input_test/nor_mirror_input_batch'
# output_path = 'dataset/TCNN_16LiFi_10m_100UE/output_test/mirror_output_batch'

input_path = 'dataset/TCNN_16LiFi_10m_100UE/input_test_new/input_batch'
output_path = 'dataset/TCNN_16LiFi_10m_100UE/output_test_new/output_batch'

input_path_list = []
output_path_list = []

for i in range(2):
    input_path_list.append(input_path+'%s.csv'%(str(i+1919)))
    output_path_list.append(output_path+'%s.csv'%str((i+1919)))

acc_list = []
for i in range(2):
    UE_list = [100]*11
    # UE_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    
    input_path = input_path_list[i]
    output_path = output_path_list[i]
    UE_num = UE_list[i]
    AP_num = 17
    # acc = test_acc_ATCNN(input_path, output_path, AP_num, UE_num)
    acc = test_acc_ATCNN_raw(M, input_path, output_path, AP_num, UE_num, 1, SNR_min)
    acc_list.append(acc)
    print('UE number is:', UE_num, 'and Accuracy is:', acc)
 
print('Aver Acc is:', sum(acc_list)/len(acc_list))

    
        
        
    
    
    
