import numpy as np

x_data = [3.90, 15.10, 0.33, 12.28, 2.33, 11.92, 10.48, 10.00, 13.60, 19.14, 11.74, 11.59, 18.02]

y_data = [9.09, 17.90, 11.47, 0.34, 15.85, 13.10, 10.67, 19.27, 7.89, 8.53, 8.43, 2.67, 10.56]

sum_x = 0
sum_y = 0

for i in x_data:
    sum_x += i

for i in y_data:
    sum_y += i

# 2
x2 = sum_x / 13.0
y2 = sum_y / 13.0

# x = 0.33
x = x2 - 3
k = 0

sumsum = []

while x <= x2 + 3:
    y = y2 - 3
    while y <= y2 + 3:
        sum2 = 0
        i = 0
        while i < 13:
            sum2 += np.sqrt(((x_data[i] - x) ** 2 + (y_data[i] - y) ** 2))
            i += 1
        # sumsum[k] = sum2
        sumsum.append(sum2)
        print(sumsum[k], "  k: ", k)
        k += 1
        y += 0.01
    x += 0.01

min_value = min(sumsum)
print("listIndex:", sumsum.index(min_value))  # 1982981
print("min_value:", min_value)  # 732.20
print()
