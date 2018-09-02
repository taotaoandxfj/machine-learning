import numpy as np

a = np.random.random((3, 4))  # 随机生成一些0到1的数据

print(a)
print(np.sum(a, axis=0))
print(np.min(a, axis=0))  # axis=0表示在每一列中寻找最小值axis=1表示在每一行中寻找最小值
print(np.max(a, axis=1))
