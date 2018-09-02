x_data = [3.90, 15.10, 0.33, 12.28, 2.33, 11.92, 10.48, 10.00, 13.60, 19.14, 11.74, 11.59, 18.02]

y_data = [9.09, 17.90, 11.47, 0.34, 15.85, 13.10, 10.67, 19.27, 7.89, 8.53, 8.43, 2.67, 10.56]

sum_xx = 0
sum_yy = 0

sum_x = 0
sum_y = 0

for i in x_data:
    sum_xx = i * i + sum_xx
    sum_x += i

for i in y_data:
    sum_yy = i * i + sum_yy
    sum_y += i

# 1
x1 = sum_xx / sum_x
y1 = sum_yy / sum_y

# 2
x2 = sum_x / 13.0
y2 = sum_y / 13.0

sum = 0

for i in range(13):
    # sum += ((x1 - x_data[i]) * (x1 - x_data[i]) + (y1 - y_data[i]) * (y1 - y_data[i]))
    sum += ((x_data[i] - 12.35) ** 2 + (y_data[i] - 10.35) ** 2)
    # print((x_data[i]-x1)**2)
    # print((y_data[i]-y1)**2)
sum1 = 0

for i in range(13):
    # sum1 += ((x2 - x_data[i]) * (x2 - x_data[i]) + (y2 - y_data[i]) * (y2 - y_data[i]))
    sum1 += ((x_data[i] - x2) ** 2 + (y_data[i] - y2) ** 2)

# print(sum_yy)
# print(sum_y)
# sum+=((x_data[i]-x1)**2+(y_data[i]-y1)**2)

# print(sum_xx / sum_x)
# print(sum_yy / sum_y)
# #
# print(sum_x / 13)
# print(sum_y / 13)

print("sum", sum)
print("sum1", sum1)

x = 0.33
k = 0
sumsum = []
# for i in range(1882 * 1894 + 1):
#     sumsum.append(0)
#
# print(len(sumsum))

# while x <= 19.14:
#     y = 0.34
#     while y <= 19.27:
#         sum2 = 0
#         i = 0
#         while i < 13:
#             sum2 += ((x_data[i] - x) ** 2 + (y_data[i] - y) ** 2)
#             i += 1
#         # sumsum[k] = sum2
#         sumsum.append(sum2)
#         print(sumsum[k], "  k: ", k)
#         k += 1
#         y += 0.01
#     x += 0.01
#
# min_value = min(sumsum)
# print("listIndex:", sumsum.index(min_value))  # 1982981
# print("min_value:", min_value)  # 732.20
# print()

# for i in range(2000 * 2000 + 1):
#     if i % 100 == 0:
#         print(sumsum[i], " ")
