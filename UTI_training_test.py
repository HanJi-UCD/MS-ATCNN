# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:06:41 2023

@author: Han
"""
'''
import math
import pandas as pd
import os
import torch.nn as nn
from ATCNN_model import MSNN
import argparse
import datetime
import json
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, TensorDataset, DataLoader


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
parser.add_argument('--exp_name', default='Description: MSNN-based UTI regression task in MS-ATCNN')
parser.add_argument('--input_dim', default=3, type=int, help='dim of input MSNN model, including [SINR, Theta, Speed]')
parser.add_argument('--output_dim', default=1, type=int, help='dim of UTI')
parser.add_argument('--batch-size', default=100, type=int, help='training and test batch size')
parser.add_argument('--test_freq', default=20, type=int)
parser.add_argument('--epoch_num', default=501, type=int)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--model_name', default='MSNN')
# parser.add_argument('--dataset_root', default='./dataset/Optimal_UTI_temporal_50UE_50M/16AP_log_UTI_100UE_new_Type4.csv')
parser.add_argument('--dataset_root', default='./dataset/Optimal_UTI4/16AP_dataset_UTI_100UE_new_Type4.csv')
args = parser.parse_args()

args.folder_to_save_files = 'result/'+datetime.datetime.now().strftime("%Y-%m-%d-%I-%M-%S-%p-")+args.model_name+'-Type4'

if not os.path.exists(args.folder_to_save_files):
    os.mkdir(args.folder_to_save_files)
    
arg_config_path = os.path.join(args.folder_to_save_files, 'Hyper-parameters.json')
with open (arg_config_path, 'w') as file:
    json.dump(vars(args), file, indent=4)

############ load training dataset  ############
raw_data = pd.read_csv(args.dataset_root)
raw_data = raw_data[raw_data['V_Tar(m/s)'] != 0]

input_features = raw_data[['SNR(dB)', 'direction_theta(rad)', 'V_Tar(m/s)']].values
output_labels = raw_data['UTI(5%)'].values

input_features = torch.tensor(input_features, dtype=torch.float32)
output_labels = torch.tensor(output_labels, dtype=torch.float32)

# normalization
SINR_max = max(input_features[:, 0]).tolist()
SINR_min = min(input_features[:, 0]).tolist()
input_features[:, 0] = (input_features[:, 0] - SINR_min)/(SINR_max - SINR_min)
input_features[:, 1] = (input_features[:, 1])/math.pi
input_features[:, 2] = (input_features[:, 2])/10
output_labels = output_labels/2
print('Max and Min SINR values are %s and %s'%(SINR_max, SINR_min))
print('')
# new RWP dataset parameters
# Type1: max 20.90, min 10.53, sample size:1024
# Type2: max 23.019, min 5.191, sample size:2584
# Type3: max 23.20, min 9.524, sample size:647
# Type3: max 65.328, min 22.834, sample size:5746

length = int(len(input_features)/100)*100

input_features = input_features[0:length,]
output_labels = output_labels[0:length,]

dataset = TensorDataset(input_features, output_labels)

train_size = int(0.8 * len(input_features))
val_size = len(input_features) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


model = MSNN(input_dim=3, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=10, gamma=1)

logger = TrainLogger(args.folder_to_save_files)

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model = model.to(device)

############ training ############ 
for epoch in range(args.epoch_num):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        
        pred_UTI = model(inputs)

        loss = criterion(pred_UTI.squeeze(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)
    scheduler.step() # update learning rate

    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_mape = 0.0
    if epoch % args.test_freq == 0:
        with torch.no_grad():
            for inputs, labels in val_loader:
                
                pred_UTI = model(inputs).squeeze()
                loss = criterion(pred_UTI.squeeze(), labels)
                
                val_loss += loss.item()
                
                # calculate loss
                abs_diff = torch.abs(pred_UTI.squeeze() - labels)
                val_mae += torch.sum(abs_diff).item()
                val_mape += torch.sum(abs_diff / labels).item()
                
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

'''
#%% test samples, compare UTI error for RNN and MSNN
import math
import pandas as pd
import torch.nn as nn
from ATCNN_model import MSNN
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

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

saved_mode_path1 = './result/temporal_UTI/2024-07-04-06-19-25-PM-MSNN-Type1/final_model.pth'
saved_mode_path2 = './result/temporal_UTI/2024-07-04-06-21-34-PM-MSNN-Type2/final_model.pth'
saved_mode_path3 = './result/temporal_UTI/2024-07-04-06-22-58-PM-MSNN-Type3/final_model.pth'
saved_mode_path4 = './result/temporal_UTI/2024-07-04-06-23-56-PM-MSNN-Type4/final_model.pth'
RNN_model_path = './result/temporal_UTI/2024-07-04-10-22-18-PM-RNN/final_model.pth'
#RNN_model_path = './result/temporal_UTI/2024-06-29-09-14-57-PM/final_model.pth' # trained RNN model on RWP model (size 10000) with AP type as input
#RNN_model_path = './result/temporal_UTI/2024-06-30-06-37-05-PM-RNNMarkov/final_model.pth' # trained RNN model on Markov mobility dataset (size 1000)

########### load time series test dataset  ############
file1 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M_new/16AP_log_UTI_100UE_new_Type1.csv')
file2 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M_new/16AP_log_UTI_100UE_new_Type2.csv')
file3 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M_new/16AP_log_UTI_100UE_new_Type3.csv')
file4 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M_new/16AP_log_UTI_100UE_new_Type4.csv')
# file1 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M_new/16AP_log_UTI_100UE_RandomWalk_Type1.csv')
# file2 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M_new/16AP_log_UTI_100UE_RandomWalk_Type2.csv')
# file3 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M_new/16AP_log_UTI_100UE_RandomWalk_Type3.csv')
# file4 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M_new/16AP_log_UTI_100UE_RandomWalk_Type4.csv')
# file1 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M/16AP_log_UTI_100UE_Markov_Type1.csv')
# file2 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M/16AP_log_UTI_100UE_Markov_Type2.csv')
# file3 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M/16AP_log_UTI_100UE_Markov_Type3.csv')
# file4 = pd.read_csv('./dataset/Optimal_UTI_temporal_50UE_50M/16AP_log_UTI_100UE_Markov_Type4.csv')
# file1['AP Type'] = 0
# file2['AP Type'] = 1
# file3['AP Type'] = 2
# file4['AP Type'] = 3

# normalization values for raw training dataset
# Type1: max 20.90, min 10.53, sample size:1024
# Type2: max 23.019, min 5.191, sample size:2584
# Type3: max 23.20, min 9.524, sample size:647
# Type4: max 65.328, min 22.834, sample size:5746

file1 = file1[file1['V_Tar(m/s)'] != 0]
file2 = file2[file2['V_Tar(m/s)'] != 0]
file3 = file3[file3['V_Tar(m/s)'] != 0]
file4 = file4[file4['V_Tar(m/s)'] != 0]
combined_data = pd.concat([file1, file2, file3, file4]).sort_values(by='Test')

combined_data = combined_data[combined_data['V_Tar(m/s)'] != 0]

input_features = combined_data[['SNR(dB)', 'direction_theta(rad)', 'V_Tar(m/s)']].values
output_labels = combined_data['UTI(5%)'].values

input_features = torch.tensor(input_features, dtype=torch.float32)
output_labels = torch.tensor(output_labels, dtype=torch.float32)

# normalization
# SINR_max = max(input_features[:, 0]).tolist()
# SINR_min = min(input_features[:, 0]).tolist()
SINR_max = 65.328
SINR_min = 5.191

input_features[:, 0] = (input_features[:, 0] - SINR_min)/(SINR_max - SINR_min)
input_features[:, 1] = (input_features[:, 1])/math.pi
input_features[:, 2] = (input_features[:, 2])/10
print('Max and Min SINR values are %s and %s'%(SINR_max, SINR_min))
print('')

dataset = SequenceDataset(input_features, output_labels, 10)

length = int(len(dataset)/10)*10
trainng_dataset= torch.utils.data.Subset(dataset, range(length)) #
test_loader = DataLoader(trainng_dataset, batch_size=10, shuffle=False)

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
# load RNN model
model_RNN = RNN(3, 32, 1, 3, 0.5)
model_RNN.load_state_dict(torch.load(RNN_model_path, map_location=device), strict=False)
model_RNN.eval()
model_RNN.to(device)

RNN_error = []
h0 = torch.zeros(3, 10, 32).to(device)
for inputs, labels in test_loader:
    # RNN model
    outputs, h0 = model_RNN(inputs, h0)
    RNN_pred_UTI = outputs.squeeze()*2
    error_now = RNN_pred_UTI - labels.squeeze()
    RNN_error.append(error_now.tolist())



############ load MSNN model and dataset  ############
###### Type I UTI ######
input_features = file1[['SNR(dB)', 'direction_theta(rad)', 'V_Tar(m/s)']].values
output_labels = file1['UTI(5%)'].values
input_features = torch.tensor(input_features, dtype=torch.float32)
output_labels = torch.tensor(output_labels, dtype=torch.float32)
# normalization
# SINR_max = max(input_features[:, 0]).tolist()
# SINR_min = min(input_features[:, 0]).tolist()
SINR_max = 20.90
SINR_min = 10.53
input_features[:, 0] = (input_features[:, 0] - SINR_min)/(SINR_max - SINR_min)
input_features[:, 1] = (input_features[:, 1])/math.pi
input_features[:, 2] = (input_features[:, 2])/10

length = int(len(file1)/10)*10
input_features = input_features[0:length,]
output_labels = output_labels[0:length,]

dataset = TensorDataset(input_features, output_labels)
test_loader = DataLoader(dataset, batch_size=10, shuffle=False)

model_MSNN = MSNN(input_dim=3, output_dim=1)
device = torch.device('cpu')
model_MSNN.load_state_dict(torch.load(saved_mode_path1, map_location=device), strict=False)
model_MSNN.eval()
model_MSNN.to(device)

MSNN_error1 = []
for inputs, labels in test_loader:
    # RNN model
    outputs = model_MSNN(inputs)
    MSNN_pred_UTI = outputs.squeeze()*2
    error_now = MSNN_pred_UTI - labels.squeeze()
    MSNN_error1.append(error_now.tolist())


###### Type II UTI ######
input_features = file2[['SNR(dB)', 'direction_theta(rad)', 'V_Tar(m/s)']].values
output_labels = file2['UTI(5%)'].values
input_features = torch.tensor(input_features, dtype=torch.float32)
output_labels = torch.tensor(output_labels, dtype=torch.float32)
# normalization
# SINR_max = max(input_features[:, 0]).tolist()
# SINR_min = min(input_features[:, 0]).tolist()
SINR_max = 23.019
SINR_min = 5.191
input_features[:, 0] = (input_features[:, 0] - SINR_min)/(SINR_max - SINR_min)
input_features[:, 1] = (input_features[:, 1])/math.pi
input_features[:, 2] = (input_features[:, 2])/10

length = int(len(file2)/10)*10
input_features = input_features[0:length,]
output_labels = output_labels[0:length,]

dataset = TensorDataset(input_features, output_labels)
test_loader = DataLoader(dataset, batch_size=10, shuffle=False)

model_MSNN = MSNN(input_dim=3, output_dim=1)
device = torch.device('cpu')
model_MSNN.load_state_dict(torch.load(saved_mode_path2, map_location=device), strict=False)
model_MSNN.eval()
model_MSNN.to(device)

MSNN_error2 = []
for inputs, labels in test_loader:
    # RNN model
    outputs = model_MSNN(inputs)
    MSNN_pred_UTI = outputs.squeeze()*2
    error_now = MSNN_pred_UTI - labels.squeeze()
    MSNN_error2.append(error_now.tolist())

###### Type III UTI ######
input_features = file3[['SNR(dB)', 'direction_theta(rad)', 'V_Tar(m/s)']].values
output_labels = file3['UTI(5%)'].values
input_features = torch.tensor(input_features, dtype=torch.float32)
output_labels = torch.tensor(output_labels, dtype=torch.float32)
# # normalization
# SINR_max = max(input_features[:, 0]).tolist()
# SINR_min = min(input_features[:, 0]).tolist()
SINR_max = 23.2
SINR_min = 9.524
input_features[:, 0] = (input_features[:, 0] - SINR_min)/(SINR_max - SINR_min)
input_features[:, 1] = (input_features[:, 1])/math.pi
input_features[:, 2] = (input_features[:, 2])/10

length = int(len(file3)/10)*10
input_features = input_features[0:length,]
output_labels = output_labels[0:length,]

dataset = TensorDataset(input_features, output_labels)
test_loader = DataLoader(dataset, batch_size=10, shuffle=False)

model_MSNN = MSNN(input_dim=3, output_dim=1)
device = torch.device('cpu')
model_MSNN.load_state_dict(torch.load(saved_mode_path3, map_location=device), strict=False)
model_MSNN.eval()
model_MSNN.to(device)

MSNN_error3 = []
for inputs, labels in test_loader:
    # RNN model
    outputs = model_MSNN(inputs)
    MSNN_pred_UTI = outputs.squeeze()*2
    error_now = MSNN_pred_UTI - labels.squeeze()
    MSNN_error3.append(error_now.tolist())
    
    
###### Type IV UTI ######
input_features = file4[['SNR(dB)', 'direction_theta(rad)', 'V_Tar(m/s)']].values
output_labels = file4['UTI(5%)'].values
input_features = torch.tensor(input_features, dtype=torch.float32)
output_labels = torch.tensor(output_labels, dtype=torch.float32)
# normalization
# SINR_max = max(input_features[:, 0]).tolist()
# SINR_min = min(input_features[:, 0]).tolist()
SINR_max = 65.328
SINR_min = 22.834
input_features[:, 0] = (input_features[:, 0] - SINR_min)/(SINR_max - SINR_min)
input_features[:, 1] = (input_features[:, 1])/math.pi
input_features[:, 2] = (input_features[:, 2])/10

length = int(len(file4)/10)*10
input_features = input_features[0:length,]
output_labels = output_labels[0:length,]

dataset = TensorDataset(input_features, output_labels)
test_loader = DataLoader(dataset, batch_size=10, shuffle=False)

model_MSNN = MSNN(input_dim=3, output_dim=1)
device = torch.device('cpu')
model_MSNN.load_state_dict(torch.load(saved_mode_path4, map_location=device), strict=False)
model_MSNN.eval()
model_MSNN.to(device)

MSNN_error4 = []
for inputs, labels in test_loader:
    # RNN model
    outputs = model_MSNN(inputs)
    MSNN_pred_UTI = outputs.squeeze()*2
    error_now = MSNN_pred_UTI - labels.squeeze()
    MSNN_error4.append(error_now.tolist())
    
    
# save test errors
RNN_flat_list = [item for sublist in RNN_error for item in sublist]

MSNN_flat_list1 = [item for sublist in MSNN_error1 for item in sublist]
MSNN_flat_list2 = [item for sublist in MSNN_error2 for item in sublist]
MSNN_flat_list3 = [item for sublist in MSNN_error3 for item in sublist]
MSNN_flat_list4 = [item for sublist in MSNN_error4 for item in sublist]
MSNN_error = MSNN_flat_list1 + MSNN_flat_list2 + MSNN_flat_list3 + MSNN_flat_list4

max_length = max(len(RNN_flat_list), len(MSNN_error))

RNN_flat_list += [None] * (max_length - len(RNN_flat_list))
MSNN_error += [None] * (max_length - len(MSNN_error))

data = {
    'RNN error': RNN_flat_list,
    'MSNN error': MSNN_error
}
df = pd.DataFrame(data)
df.to_csv('./result/temporal_UTI/RNN_MSNN_UTI_error_final_RandomWalk.csv', index=False)











