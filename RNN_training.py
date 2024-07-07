#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 20:26:17 2024

@author: han
"""

#%% data preprocessing
import pandas as pd
import torch
import math
import os
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import datetime
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

############  load dataset ############
file1 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M/16AP_log_UTI_100UE_new_Type1.csv')
file2 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M/16AP_log_UTI_100UE_new_Type2.csv')
file3 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M/16AP_log_UTI_100UE_new_Type3.csv')
file4 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M/16AP_log_UTI_100UE_new_Type4.csv')
# file1 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M/16AP_log_UTI_100UE_Markov_Type1.csv')
# file2 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M/16AP_log_UTI_100UE_Markov_Type2.csv')
# file3 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M/16AP_log_UTI_100UE_Markov_Type3.csv')
# file4 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M/16AP_log_UTI_100UE_Markov_Type4.csv')
# add another input of AP type in RNN
# file1['AP Type'] = 0
# file2['AP Type'] = 1
# file3['AP Type'] = 2
# file4['AP Type'] = 3

combined_data = pd.concat([file1, file2, file3, file4]).sort_values(by='Test')

input_features = combined_data[['SNR(dB)', 'direction_theta(rad)', 'V_Tar(m/s)']]
output_labels = combined_data['UTI(5%)']

features_numpy = input_features.values
labels_numpy = output_labels.values

input_features = torch.tensor(features_numpy, dtype=torch.float32)
output_labels = torch.tensor(labels_numpy, dtype=torch.float32)

# input normalization
SINR_max = max(input_features[:, 0]).tolist()
SINR_min = min(input_features[:, 0]).tolist()
input_features[:, 0] = (input_features[:, 0] - SINR_min)/(SINR_max - SINR_min)
input_features[:, 1] = (input_features[:, 1])/math.pi
input_features[:, 2] = (input_features[:, 2])/10
output_labels = output_labels/2
print('Max and Min SINR values are %s and %s'%(SINR_max, SINR_min))
print('')

class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_length):
        self.X = X
        self.y = y
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_length]
        y_label = self.y[idx + self.seq_length - 1]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.float32)
    
#  RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout = 0.0):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, h0):
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.Sigmoid(out)
        return out, hn
    
class TrainLogger():
    def __init__(self, path, title=None):
        if title == None:
            self.title = ['Epoch', 'Training loss', 'Test loss', 'MAE', 'MAPE']
        else:
            self.title = title
        self.data = []
        self.path = path
    
    def update(self, log):
        self.data.append(log)
        df = pd.DataFrame(data=self.data, columns=self.title)
        df.to_csv(self.path + '/log.csv', index=False, sep=',')
        
    def plot(self):
        #####
        data = pd.read_csv(self.path + '/log.csv')
        plt.figure(figsize=(6,6))
        # MSE Plot
        plt.plot(data['Epoch'], data['Training loss'], label='Training loss')
        plt.plot(data['Epoch'], data['Test loss'], label='Validation loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.tight_layout()
        plt.grid()
        plt.savefig(f'{self.path}/training_loss.png')
        plt.close()
    
############ define training parameters ############
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='Description: RNN-based UTI regression task in MS-ATCNN')
parser.add_argument('--input_dim', default=3, type=int, help='dim of input MSNN model, including [SINR, Theta, Speed]')
parser.add_argument('--hidden_dim', default=32, type=int, help='dim of hidden layers')
parser.add_argument('--output_dim', default=1, type=int, help='dim of UTI')
parser.add_argument('--layer_num', default=3, type=int, help=' number of RNN layers')
parser.add_argument('--batch-size', default=100, type=int, help='training and test batch size')
parser.add_argument('--seq_size', default=10, type=int, help='RNN input sequence length')
parser.add_argument('--test_freq', default=10, type=int)
parser.add_argument('--epoch_num', default=301, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--dropout', default=0.5, type=int)
parser.add_argument('--model_name', default='RNN')
args = parser.parse_args()

args.folder_to_save_files = 'result/temporal_UTI/'+datetime.datetime.now().strftime("%Y-%m-%d-%I-%M-%S-%p-")+args.model_name

if not os.path.exists(args.folder_to_save_files):
    os.mkdir(args.folder_to_save_files)
    
arg_config_path = os.path.join(args.folder_to_save_files, 'Hyper-parameters.json')
with open (arg_config_path, 'w') as file:
    json.dump(vars(args), file, indent=4)

dataset = SequenceDataset(input_features, output_labels, args.seq_size)

trainng_dataset= torch.utils.data.Subset(dataset, range(800))
test_dataset= torch.utils.data.Subset(dataset, range(800, 1000))

train_loader = DataLoader(trainng_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

model = RNN(args.input_dim, args.hidden_dim, args.output_dim, args.layer_num, dropout=args.dropout)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = model.to(device)

logger = TrainLogger(args.folder_to_save_files)

############ training ############ 
h0 = torch.zeros(args.layer_num, args.batch_size, args.hidden_dim).to(device)
for epoch in range(args.epoch_num):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs, h0 = model(inputs, h0)
        
        h0 = h0.detach()

        loss = criterion(outputs.squeeze(), targets.squeeze())  
        loss.backward()  
        optimizer.step()  

        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)
    scheduler.step() # update learning rate

    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_mape = 0.0
    if epoch % args.test_freq == 0:
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
    
                h0 = torch.zeros(args.layer_num, inputs.size(0), args.hidden_dim).to(device)
    
                outputs, h0 = model(inputs, h0)
                loss = criterion(outputs.squeeze(), targets.squeeze())
                val_loss += loss.item() * inputs.size(0)
                
                # calculate loss
                abs_diff = torch.abs(outputs.squeeze() - targets.squeeze())
                val_mae += torch.sum(abs_diff).item()
                val_mape += torch.sum(abs_diff / targets.squeeze()).item()
                
            val_loss /= len(val_loader.dataset)
            val_mae /= len(val_loader.dataset)
            val_mape /= len(val_loader.dataset)
        
            log = [epoch, train_loss, val_loss, val_mae, val_mape]
            logger.update(log)
        print('')
        print('*'*100)
        print(f'Learning Rate: {scheduler.get_lr()[0]:.5f}')
        print(f'Epoch {epoch}:, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE Loss: {val_mae:.4f}, Val MAPE Loss: {val_mape:.4f}')
        print('Running at' + datetime.datetime.now().strftime(" %Y_%m_%d-%I_%M_%S_%p"))

save_model_path = args.folder_to_save_files 
path = save_model_path + '/final_model.pth'
torch.save(model.state_dict(), path)
logger.plot()
print('Training finished.')

#%%
# test the trained RNN model trained
# predicted UTI versus real UTI
import torch
from torch.utils.data import DataLoader

# load trained model
model_path = 'result/temporal_UTI/2024-06-24-06-55-16-PM/'+'final_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

batch_size = 100
num_layers = 3
hidden_size = 32

test_dataset = torch.utils.data.Subset(dataset, range(7000, 9900))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_mae = 0.0
test_mape = 0.0
test_loss = 0.0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        h0 = torch.zeros(num_layers, inputs.size(0), hidden_size).to(device)

        outputs, h0 = model(inputs, h0)
        loss = criterion(outputs.squeeze(), targets.squeeze())
        test_loss += loss.item() * inputs.size(0)

        # MAE MAPE
        abs_diff = torch.abs(outputs.squeeze() - targets.squeeze())
        test_mae += torch.sum(abs_diff).item()
        test_mape += torch.sum(abs_diff / targets.squeeze()).item()

test_loss /= len(test_loader.dataset)
test_mae /= len(test_loader.dataset)
test_mape /= len(test_loader.dataset)

print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test MAPE: {test_mape:.4f}')












