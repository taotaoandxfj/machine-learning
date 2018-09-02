import numpy as np

# 矩阵的切割操作

a = np.arange(12).reshape(3, 4)
print(a)
# 对矩阵进行切割 等量

print(np.split(a, 2, axis=1))  # 把行切成两份
print(np.split(a, 3, axis=0))  # 把行切成三份

# 对矩阵进行不等量的分割

print(np.array_split(a, 2, axis=0))

# 进行纵向的分割

print(np.vsplit(a, 3))

# 进行横向的分割

print(np.hsplit(a, 2))
s