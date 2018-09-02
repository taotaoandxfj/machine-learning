import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

x_data = [3.90, 15.10, 0.33, 12.28, 2.33, 11.92, 10.48, 10.00, 13.60, 19.14, 11.74, 11.59, 18.02, 12.35]

y_data = [9.09, 17.90, 11.47, 0.34, 15.85, 13.10, 10.67, 19.27, 7.89, 8.53, 8.43, 2.67, 10.56, 10.53]

value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

print(len(x_data))
print(len(y_data))

c_value = ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'b']

plt.scatter(x_data, y_data, c=c_value)

plt.annotate("center", xy=(12.35, 10.53), xytext=(12.35 + 0.2, 10.53 + 0.2))

for x, y, v in zip(x_data, y_data, value):
    plt.annotate("%s" % v, xy=(x, y), xytext=(x + 0.2, y + 0.2))
plt.show()
