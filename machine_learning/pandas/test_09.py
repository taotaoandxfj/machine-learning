# 配送中心到各个点的时间


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

x = 12.35
y = 10.53
x_data = [3.90, 15.10, 0.33, 12.28, 2.33, 11.92, 10.48, 10.00, 13.60, 19.14, 11.74, 11.59, 18.02, 12.35]

y_data = [9.09, 17.90, 11.47, 0.34, 15.85, 13.10, 10.67, 19.27, 7.89, 8.53, 8.43, 2.67, 10.56, 10.53]

id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
sum_d = []
for i in range(13):
    sum_d.append(np.sqrt((x - x_data[i]) ** 2 + (y - y_data[i]) ** 2))

# print(sum(sum_d))

sum_f = np.array(sum_d).reshape(13)
for i in range(13):
    sum_f[i] = round(sum_f[i], 2)

print(sum_f)

c_value = ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'b']

plt.scatter(x_data, y_data, c=c_value)

plt.annotate("center", xy=(12.35, 10.53), xytext=(12.35 + 0.2, 10.53 + 0.2))

for x, y, v, i in zip(x_data, y_data, sum_f, id):
    plt.annotate("(%s,%s)" % (i, v), xy=(x, y), xytext=(x + 0.2, y + 0.3))
plt.show()
