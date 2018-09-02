import numpy as np
from decimal import Decimal

x = 12.35
y = 10.53
x_data = [3.90, 15.10, 0.33, 12.28, 2.33, 11.92, 10.48, 10.00, 13.60, 19.14, 11.74, 11.59, 18.02]

y_data = [9.09, 17.90, 11.47, 0.34, 15.85, 13.10, 10.67, 19.27, 7.89, 8.53, 8.43, 2.67, 10.56]
sum_d = []
for i in range(13):
    sum_d.append(np.sqrt((x - x_data[i]) ** 2 + (y - y_data[i]) ** 2))
    # print(x - x_data[i])
print(sum_d)

for i in range(13):
    sum_d[i] = Decimal(str(sum_d[i])).quantize(Decimal('0.00'))

print(sum_d)