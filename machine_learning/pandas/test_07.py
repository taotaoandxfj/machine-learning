# 用户1到13的时间窗上界
import operator

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

up_time = {'1': 137, '2': 124, '3': 297, '4': 172, '5': 55, '6': 148, '7': 184, '8': 148, '9': 148, '10': 247, '11': 24,
           '12': 211, '13': 69
           }

bottom_time = {'1': 169, '2': 154, '3': 327, '4': 202, '5': 85, '6': 178, '7': 214, '8': 178, '9': 174, '10': 277,
               '11': 54,
               '12': 241, '13': 99
               }

x_data = [3.90, 15.10, 0.33, 12.28, 2.33, 11.92, 10.48, 10.00, 13.60, 19.14, 11.74, 11.59, 18.02, 12.35]

y_data = [9.09, 17.90, 11.47, 0.34, 15.85, 13.10, 10.67, 19.27, 7.89, 8.53, 8.43, 2.67, 10.56, 10.53]

value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
up_time_list = []
bottom_time_list = []
for i, j in zip(up_time.values(), bottom_time.values()):
    up_time_list.append(i)
    bottom_time_list.append(j)

print(up_time_list)
print(bottom_time_list)

# 进行时间轴的排序

data = sorted(up_time.items(), key=operator.itemgetter(1))
# data1 = sorted(bottom_time.items(), key=operator.itemgetter(1))
print(data)
# print(data1)
c_value = ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'b']

plt.scatter(x_data, y_data, c=c_value)
plt.annotate("center", xy=(12.35, 10.53), xytext=(12.35 + 0.2, 10.53 + 0.2))

for x, y, v, i in zip(x_data, y_data, value, up_time.values()):
    plt.annotate("(%s,%s)" % (v, i), xy=(x, y), xytext=(x + 0.2, y + 0.3))
plt.show()
