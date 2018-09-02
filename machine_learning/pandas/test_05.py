import numpy as np
import pandas as pd

x_data = [3.90, 15.10, 0.33, 12.28, 2.33, 11.92, 10.48, 10.00, 13.60, 19.14, 11.74, 11.59, 18.02]

y_data = [9.09, 17.90, 11.47, 0.34, 15.85, 13.10, 10.67, 19.27, 7.89, 8.53, 8.43, 2.67, 10.56]

weights = [60, 50, 61, 35, 14, 27, 23, 36, 49, 42, 32, 17, 54]

# print(len(weights))

#  x
sum_x = 0
sum_y = 0
x = 0
y = 0

for i in range(13):
    sum_x += x_data[i] * weights[i]
    sum_y += y_data[i] * weights[i]

x = sum_x / sum(weights)
y = sum_y / sum(weights)

print("x:", x, "  ", "y:", y)
