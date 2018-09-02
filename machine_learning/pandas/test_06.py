import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_data = [3.90, 15.10, 0.33, 12.28, 2.33, 11.92, 10.48, 10.00, 13.60, 19.14, 11.74, 11.59, 18.02, 9.56, 6.44, 10.35]

y_data = [9.09, 17.90, 11.47, 0.34, 15.85, 13.10, 10.67, 19.27, 7.89, 8.53, 8.43, 2.67, 10.56, 6.03, 11.28, 5.27]

value = [60, 50, 61, 35, 14, 27, 23, 36, 49, 42, 32, 17, 54]

c_value = ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'b', 'b', 'b']

plt.scatter(x_data, y_data, c=c_value)

plt.annotate("center1", xy=(9.56, 6.03), xytext=(9.56 + 0.2, 6.03 + 0.2))
plt.annotate("center2", xy=(6.44, 11.28), xytext=(6.44 + 0.2, 11.28 + 0.2))
plt.annotate("center3", xy=(10.35, 5.27), xytext=(10.35 + 0.2, 5.27 + 0.2))

for x, y, v in zip(x_data, y_data, value):
    plt.annotate("%s" % v, xy=(x, y), xytext=(x + 0.2, y + 0.2))
plt.show()
