import numpy as np

a = np.arange(3, 15).reshape((3, 4))

print(a)
# 迭代行
for colume in a:
    print(colume)

# 迭代列

for row in a.T:
    print(row)

# 迭代矩阵中的每一个元素

for item in a.flat:  # a.flat会将转换为1行的序列
    print(item)

# 矩阵的合并
a = np.array([1, 1, 1])
b = np.array([2, 2, 2])

# 按上下来进行合并

c = np.vstack((a, b))
print(c)
# 按左右来进行合并
d = np.hstack((a, b))
print(d)

# 将一个序列变得来有维度

# 就a来说是一个数组并不是一个矩阵 如何变成矩阵的形式呢? 很显然你直接转置时不行的

# 那么可以用如下的方式来进行变换 当然也可以用reshape方法

print(a.shape, a.T)
'''
[[1]
 [1]
 [1]]
'''
print(a[:, np.newaxis], a.shape)
print(a[np.newaxis, :], a.shape)  # [[1 1 1]]

# 按指定的方式来进行合并
a = np.array([1, 1, 1])[:, np.newaxis]
b = np.array([2, 2, 2])[:, np.newaxis]
e = np.concatenate((a, b, b, a), axis=1)
print("="*10)
print(e)
