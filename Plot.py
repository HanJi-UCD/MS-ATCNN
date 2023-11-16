# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:43:37 2023

@author: Han
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件数据
file_path = 'result/Fig1_Thr_vs_UTI_UE70.csv'  # 将文件路径替换为实际路径
data = pd.read_csv(file_path)

# 提取V_Tar和第6到10列数据
v_tar = data['V_Tar(m/s)']
columns_6_to_10 = data.iloc[:, 5:10]

# 获取不重复的速度值和分组数
unique_v_tar = sorted(v_tar.unique())
num_groups = len(unique_v_tar)

# 计算每组数据的平均值
grouped_means = columns_6_to_10.groupby(v_tar).mean()

# 创建柱状图
plt.figure(figsize=(12, 6))

# 循环遍历每个速度并绘制柱状图
bar_width = 0.2
x_positions = np.arange(0, num_groups * 5 * (bar_width + 0.1), 5 * (bar_width + 0.1))

for idx, v in enumerate(unique_v_tar):
    v_means = grouped_means.loc[v].values
    plt.bar(x_positions + idx * bar_width, v_means, width=bar_width, label=f'Speed {v}')


plt.xlabel('Group')
plt.ylabel('Average Data')
plt.title('Bar Chart for Grouped Averages')
plt.xticks(x_positions + bar_width * (num_groups - 1) / 2, range(1, num_groups * 5 + 1, 5))
plt.legend()
plt.tight_layout()
plt.show()