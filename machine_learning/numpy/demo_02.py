import numpy as np

a = np.array([[1, 1], [2, 2]])
b = np.array([[3, 3], [4, 4]])

# 矩阵的加减乘
print(a + b)

print(a - b)

print(a ** 2)

print(b ** 2)

print(b > 3)  # [[False False]  [ True  True]]

print(b == 3)

print(a * b)  # 表示元素逐个相乘

print(np.dot(a, b))

