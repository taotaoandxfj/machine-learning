import numpy as np
import pandas as pd

x_data = [3.90, 15.10, 0.33, 12.28, 2.33, 11.92, 10.48, 10.00, 13.60, 19.14, 11.74, 11.59, 18.02]

y_data = [9.09, 17.90, 11.47, 0.34, 15.85, 13.10, 10.67, 19.27, 7.89, 8.53, 8.43, 2.67, 10.56]

x = 10.35
y = 5.27


x_data_add = []
y_data_add = []

for i in x_data:
    x_data_add.append(i)
x_data_add.append(x)

for i in y_data:
    y_data_add.append(i)
    y_data_add.append(i)
y_data_add.append(y)

sum = []
for i in range(14):
    for j in range(14):
        sum.append(np.sqrt((x_data_add[i] - x_data_add[j]) ** 2 + (y_data_add[i] - y_data_add[j]) ** 2))


data = np.array(sum).reshape((len(x_data_add), len(y_data_add)))

# print(data)
for i in range(14):
    for j in range(14):
        data[i][j] = round(data[i][j], 2)

print(data)

data_s = pd.DataFrame(data, index=np.arange(1, 15), columns=np.arange(1, 15))
data_s.to_csv("data3_s.csv")
print(data_s)
