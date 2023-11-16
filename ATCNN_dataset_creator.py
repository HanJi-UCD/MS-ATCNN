#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 19:21:35 2022

@author: han
"""
#%%

import numpy as np
import csv
from glob import glob
import torch
import os
import pickle

# Mannully set the path to the folder where storing the data
mode = 'input'
dataset_path = 'dataset/TCNN_16LiFi_10m_100UE/'

if mode == 'input':
    raw_dataset_path = dataset_path+'input_test'
    output_file_name = dataset_path+'input_test.h5'
elif mode == 'output':
    raw_dataset_path =  dataset_path+'output_test'
    output_file_name =  dataset_path+'output_test.h5'
    
paths = glob(raw_dataset_path + '/*')

for idx in range(len(paths)): 
    with open(os.path.join(raw_dataset_path, f'nor_mirror_input_batch{idx+1891}.csv'), encoding="utf-8") as f:
        reader = csv.reader(f)
        sub_dataset = []
        for row in reader:
            data_row = []
            for fig in row:
                data_row.append(float(fig.strip()))
            sub_dataset.append(data_row)
        sub_dataset = np.array(sub_dataset).transpose(1,0)

        if idx == 0:
            dataset = sub_dataset
        else:
            dataset = np.concatenate((dataset, sub_dataset), axis=0)
        if idx % 100 == 0:
            print(f'{idx}/{len(paths)} sets have been finished')

print(f'Saving to file {output_file_name}')
print('Shape of the dataset is: ', dataset.shape)

torch.save(dataset, output_file_name)
# save large model using pcikle
# with open("dataset/ATCNN_16LiFi_10m_100UE/input_test.pkl", "wb") as f:
#     pickle.dump(dataset, f)
    
#%%
import numpy as np
import csv
from glob import glob
import torch
import os
import pickle

mode = 'output'
dataset_path = 'dataset/TCNN_16LiFi_10m_100UE/'

if mode == 'input':
    raw_dataset_path = dataset_path+'input_train'
    output_file_name = dataset_path+'input_train.h5'
elif mode == 'output':
    raw_dataset_path =  dataset_path+'output_train'
    output_file_name =  dataset_path+'output_train.h5'
    
paths = glob(raw_dataset_path + '/*')

for idx in range(len(paths)): 
    with open(os.path.join(raw_dataset_path, f'mirror_output_batch{idx+1801}.csv'), encoding="utf-8") as f:
        reader = csv.reader(f)
        sub_dataset = []
        for row in reader:
            data_row = []
            for fig in row:
                data_row.append(float(fig.strip()))
            sub_dataset.append(data_row)
        sub_dataset = np.array(sub_dataset).transpose(1,0)
        if idx == 0:
            dataset = sub_dataset
        else:
            dataset = np.concatenate((dataset,sub_dataset),axis=0)
        if idx % 100 == 0:
            print(f'{idx}/{len(paths)} sets have been finished')

print(f'Saving to file {output_file_name}')
print('Shape of the dataset is: ', dataset.shape)
    
torch.save(dataset,output_file_name)
# save large model using pcikle
# with open("dataset/ATCNN_16LiFi_10m_100UE/output_train.pkl", "wb") as f:
#     pickle.dump(dataset, f, protocol=4)
    
#%%
import numpy as np
import csv
from glob import glob
import torch
import os
import pickle

mode = 'input'
dataset_path = 'dataset/ATCNN_16LiFi_10m_100UE/'

if mode == 'input':
    raw_dataset_path = dataset_path+'input_train'
    output_file_name = dataset_path+'input_train.h5'
elif mode == 'output':
    raw_dataset_path =  dataset_path+'output_train'
    output_file_name =  dataset_path+'output_train.h5'
    
paths = glob(raw_dataset_path + '/*')

for idx in range(len(paths)): 
    with open(os.path.join(raw_dataset_path, f'nor_mirror_input_batch{idx+1}.csv'), encoding="utf-8") as f:
        reader = csv.reader(f)
        sub_dataset = []
        for row in reader:
            data_row = []
            for fig in row:
                data_row.append(float(fig.strip()))
            sub_dataset.append(data_row)
        sub_dataset = np.array(sub_dataset).transpose(1,0)
        if idx == 0:
            dataset = sub_dataset
        else:
            dataset = np.concatenate((dataset,sub_dataset),axis=0)
        if idx % 100 == 0:
            print(f'{idx}/{len(paths)} sets have been finished')

print(f'Saving to file {output_file_name}')
print('Shape of the dataset is: ', dataset.shape)
    
# torch.save(dataset,output_file_name)
# save large model using pcikle
with open("dataset/ATCNN_16LiFi_10m_100UE/input_train.pkl", "wb") as f:
    pickle.dump(dataset, f, protocol=4)