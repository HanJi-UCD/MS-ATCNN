# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:14:37 2023

@author: Han
"""

###### test UTI runtime ######
###### This runtime is not ture runtime, using torch not tensorflow
import numpy as np
import torch
import time
import tensorflow as tf

loaded_model = tf.keras.models.load_model('trained_model/NN_UTI_100UE_Type1.h5')
total_time = 0
times = 100

for i in range(times):
    input_now = np.random.rand(1,4).tolist()
    tic = time.time()
    input_now = np.array(input_now)
    pre_output = loaded_model.predict(input_now).tolist()
    toc = time.time()
    total_time = total_time + (toc - tic)
print('Average runtime of UTI prediction is %s', total_time/times)
###### Average Runtime of UTI is 0.164 ms

#%%
###### Runtime for ATCNN ######
from ATCNN_model import ATCNN_16LiFi_100UE as ATCNN
import numpy as np
import torch
import matlab.engine
import json
from mytopo import Net, mobility_trace

user_dim = 18 # 
output_dim = 17  # 
cond_dim = 1800 #
AP_num = 17
M = 100

net = Net(AP_num, 5)
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

total_time = 0
count = 0

R_condition_now = 300 # randomly choose R_condition
net.waypoints_num = int(10/0.01 + 1)
net.UE_num = np.random.randint(10, 100) 
net.Velocity = [1]*net.UE_num
R_cond_raw = np.random.gamma(1, R_condition_now, net.UE_num-1)
############ generate data rate requirement in each trail ############
R_tar = 50
R_tar_now = [R_tar]
R_cond = np.clip(R_cond_raw, 10, 1000).tolist()
R_tar_now.extend(R_cond)
net.R_requirement = R_tar_now
############ update trace
net.recorded_trace = mobility_trace(net.UE_num, 10, 10, net.Velocity, 20, 10) 
net.mobility_trigger(i, net.X_list_new, net.Y_list_new, net.Saved_SNR_list) # update SNRs and Capacity

for i in range(100):
    runtime = net.load_balancing(Trained_Net, M)
    total_time = total_time + runtime
    if i%10 == 0:
        print(i)
    count = count + net.UE_num

print('Average runtime of ATCNN LB is %s', total_time/count)
###### Average Runtime of ATCNN with mapping and normalization is 2.4ms

#%%
# SSS and GT LB Runtime
import numpy as np
import torch
import matlab.engine
import json
import time
from mytopo import Net, mobility_trace

AP_num = 17
Velocity = np.linspace(1.0/100, 5.0/100, num=5).tolist() # unit of m/s: 1~5 m/s
R_aver = 100 # fix R as 100 M with Gamma distribution

net = Net(AP_num, 5)
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
runtime_list = []

for j in range(91):
    net.UE_num = 10 + j 
    total_time = 0
    ############ generate data rate requirement in each trail ############
    net.waypoints_num = int(10/0.01 + 1)
    R_now = np.random.gamma(1, R_aver, net.UE_num).tolist()
    R_requirement = np.clip(R_now, 10, 500).tolist()
    net.R_requirement = R_requirement
    
    net.Velocity = np.random.choice(Velocity, size=net.UE_num).tolist()
    ############ update trace
    net.recorded_trace = mobility_trace(net.UE_num, 10, 10, net.Velocity, 20, 10)
    net.mobility_trigger(i, net.X_list_new, net.Y_list_new, net.Saved_SNR_list) # update SNRs and Capacity
    for i in range(50):
        tic = time.time()
        net.load_balancing_GT()
        toc = time.time()
        runtime = toc-tic
        total_time = total_time + runtime
    
    runtime_list.append(total_time/50)
    print('For UE number %s, runtime is %s'%(net.UE_num, total_time/50))

# 0.042ms for SSS
################ GT, Nf = 5 #######################
# For UE number 10, runtime is 0.01517799973487854
# For UE number 11, runtime is 0.01735209584236145
# For UE number 12, runtime is 0.01946580410003662
# For UE number 13, runtime is 0.0236912441253662
# For UE number 14, runtime is 0.025754516124725343
# For UE number 15, runtime is 0.02840476989746093
# For UE number 16, runtime is 0.0312668800354
# For UE number 17, runtime is 0.033130860328674315
# For UE number 18, runtime is 0.04389487862586975
# For UE number 19, runtime is 0.05248868584632873
# For UE number 20, runtime is 0.05340155959129333
# For UE number 21, runtime is 0.06345929861068725
# For UE number 22, runtime is 0.06493329882621765
# For UE number 23, runtime is 0.06403578972816467
# For UE number 24, runtime is 0.06789239764213562
# For UE number 25, runtime is 0.07829378843307495
# For UE number 26, runtime is 0.08739854454994202
# For UE number 27, runtime is 0.09877943634986877
# For UE number 28, runtime is 0.1095826768875122
# For UE number 29, runtime is 0.11068134903907776
# For UE number 30, runtime is 0.12000594854354859
# For UE number 31, runtime is 0.1240588307380677
# For UE number 32, runtime is 0.1290666961669922
# For UE number 33, runtime is 0.14586751461029052
# For UE number 34, runtime is 0.1613818621635437
# For UE number 35, runtime is 0.16470322847366332
# For UE number 36, runtime is 0.1793335735797882
# For UE number 37, runtime is 0.18517738580703735
# For UE number 38, runtime is 0.1882637977600098
# For UE number 39, runtime is 0.2084316074848175
# For UE number 40, runtime is 0.2129347443580627
# For UE number 41, runtime is 0.23110462188720702
# For UE number 42, runtime is 0.2268852424621582
# For UE number 43, runtime is 0.25336190342903137
# For UE number 44, runtime is 0.3210641586780548
# For UE number 45, runtime is 0.2893149161338806
# For UE number 46, runtime is 0.2852368664741516
# For UE number 47, runtime is 0.32989988327026365
# For UE number 48, runtime is 0.3619178736209869
# For UE number 49, runtime is 0.35277958989143373
# For UE number 50, runtime is 0.45101271867752075
# For UE number 51, runtime is 0.47342830896377566
# For UE number 52, runtime is 0.41959859132766725
# For UE number 53, runtime is 0.5187165141105652
# For UE number 54, runtime is 0.489194929599762
# For UE number 55, runtime is 0.46372745037078855
# For UE number 56, runtime is 0.47348421812057495
# For UE number 57, runtime is 0.4753128528594971
# For UE number 58, runtime is 0.5501145601272583
# For UE number 59, runtime is 0.5608553767204285
# For UE number 60, runtime is 0.6363971352577209
# For UE number 61, runtime is 0.6545732021331788
# For UE number 62, runtime is 0.6017248463630676
# For UE number 63, runtime is 0.6363558053970337
# For UE number 64, runtime is 0.6816521739959717
# For UE number 65, runtime is 0.7094509959220886
# For UE number 66, runtime is 0.6360764861106872
# For UE number 67, runtime is 0.67245192527771
# For UE number 68, runtime is 0.7281080198287964
# For UE number 69, runtime is 0.760547935962677
# For UE number 70, runtime is 0.7301460146903992
# For UE number 71, runtime is 0.7612335205078125
# For UE number 72, runtime is 0.7668218970298767
# For UE number 73, runtime is 0.9272298097610474
# For UE number 74, runtime is 0.901051607131958
# For UE number 75, runtime is 1.015649116039276
# For UE number 76, runtime is 1.098332929611206
# For UE number 77, runtime is 0.9282322883605957
# For UE number 78, runtime is 1.068892252445221
# For UE number 79, runtime is 1.015160346031189
# For UE number 80, runtime is 1.1291860103607179
# For UE number 81, runtime is 1.1464566469192504
# For UE number 82, runtime is 1.2420321631431579
# For UE number 83, runtime is 1.0720662832260133
# For UE number 84, runtime is 1.183381187915802
# For UE number 85, runtime is 1.1218946027755737
# For UE number 86, runtime is 1.2900563240051269
# For UE number 87, runtime is 1.397345781326294
# For UE number 88, runtime is 1.3871903896331787
# For UE number 89, runtime is 1.4206083273887633
# For UE number 90, runtime is 1.456143307685852
# For UE number 91, runtime is 1.4706196784973144
# For UE number 92, runtime is 1.386559307575226
# For UE number 93, runtime is 1.41296067237854
# For UE number 94, runtime is 1.583961606025696
# For UE number 95, runtime is 1.638296902179718
# For UE number 96, runtime is 1.57868766784668
# For UE number 97, runtime is 1.6004042744636535
# For UE number 98, runtime is 1.7517176032066346
# For UE number 99, runtime is 1.8445859789848327
# For UE number 100, runtime is 1.9583533883094788
