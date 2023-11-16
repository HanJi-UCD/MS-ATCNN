# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:06:41 2023

@author: Han
"""

# simple NN training for <UE_num, R_Cond, Rho, Delay>
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import math
from keras.utils import plot_model
from mytopo import CsvLoss

class TrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0:
            print('Epoch', epoch, '- Train Loss:', logs.get('loss'), '- Test Loss:', logs.get('val_loss'))
            
def flatten_list(my_list):
    new_list = []
    for item in my_list:
        if isinstance(item, list):
            new_list.extend(flatten_list(item))
        else:
            new_list.append(item)
    return new_list

csv = CsvLoss()
############### read raw csv file and pre-process data
file_name = 'dataset/Optimal_UTI4/16AP_dataset_UTI_100UE_new_Type4.xlsx'
# choose sample number here
# df = pd.read_excel(file_name, nrows=500, dtype={'R_targetAP': 'object', 'SNR_Connected_targetAP': 'object'})
df = pd.read_excel(file_name, dtype={'R_targetAP': 'object', 'SNR_Connected_targetAP': 'object'})
df = df[df['V_Tar(m/s)']!=0] # remove 0m/s samples as theta is None 

shuffled_df = df.sample(frac=1)

df = shuffled_df[0:2000]

print('Length of training dataset is:', len(df))

df['R_targetAP'] = df['R_targetAP'].apply(lambda x: eval(x) if isinstance(x, str) else x)
df['SNR_Connected_targetAP'] = df['SNR_Connected_targetAP'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Target inputs
Tar_inputs = df.iloc[:, [4, 5, 6, 7]].values.tolist()
# Delay outputs
outputs = df.iloc[:, [16]].values.tolist()

# normalize target inputs
Tar_inputs = np.array(Tar_inputs)
max_tar_SNR= np.max(Tar_inputs[:, 0])
min_tar_SNR= np.min(Tar_inputs[:, 0])
print('Max Target SNR is %s, and min Target SNR is %s:'%(max_tar_SNR, min_tar_SNR))
Tar_inputs[:, 0] = (Tar_inputs[:, 0] - min_tar_SNR)/(max_tar_SNR - min_tar_SNR)

Tar_inputs[:, 1] = Tar_inputs[:, 1]/math.pi
Tar_inputs[:, 2] = Tar_inputs[:, 2]/500
Tar_inputs[:, 3] = Tar_inputs[:, 3]/10
Tar_inputs = Tar_inputs.tolist()

# normalize output delay
outputs = [x[0]/2 for x in outputs]

# combine input data
new_inputs = [x for x in zip(Tar_inputs)]

new_flatten_inputs = []
for i in range(len(new_inputs)):
    new_flatten_inputs.append(flatten_list(new_inputs[i]))

#
X_train, X_test, y_train, y_test = train_test_split(new_flatten_inputs, outputs, test_size=0.2, random_state=42)

# 定义神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

# 编译模型并进行训练
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=200, verbose=0, validation_data=(X_test, y_test),
                    callbacks=[tf.keras.callbacks.History(), TrainingCallback()])

train_loss = history.history['loss']
test_loss = history.history['val_loss']
print('Train Loss:', train_loss[-1])
print('Test Loss:', test_loss[-1])

plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

############
model.save('trained_model/NN_UTI_100UE_new_Type4.h5')

for i in range(200):
    log = [i, train_loss[i], test_loss[i]]
    csv.update(log, 'result/MS_Loss_Type4.csv')

#%% test samples

############# Without condition UE cases, one NN model ##################

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import math
from keras.utils import plot_model
import random

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
# df = shuffled_df[2900:3060]
# Type2
# max_tar_SNR = 23.01966378
# min_tar_SNR = 5.34853973
# df = shuffled_df[4900:5085]
# Type3
# max_tar_SNR = 23.21163127
# min_tar_SNR = 10.7606035
# df = shuffled_df[1600:1767]
# Type4
max_tar_SNR = 65.3282696916349
min_tar_SNR = -0.871427041
df = shuffled_df[6000:6200]

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
# plt.savefig('./Plots/Final_UE4_Delay_new.svg', format='svg', dpi=600)

# show errors figure
fig2 = plt.figure()
errors = [x[0] - y for x, y in zip(real_outputs, pre_output_list)]
plt.hist(errors, bins=20, color='blue', edgecolor='black')
plt.title('Error Distribution')
plt.xlabel('Error (seconds)')
plt.ylabel('Count')
plt.show()

prob0 = calc_interval_prob(errors, [-0.05, 0.05])
print('Probability of error between -50 and 50 ms is: %s'%prob0)
prob1 = calc_interval_prob(errors, [-0.1, 0.1])
print('Probability of error between -100 and 100 ms is: %s'%prob1)
prob2 = calc_interval_prob(errors, [-0.2, 0.2])
print('Probability of error between -200 and 200 ms is: %s'%prob2)
prob3 = calc_interval_prob(errors, [-0.5, 0.5])
print('Probability of error between -500 and 500 ms is: %s'%prob3)



