import numpy as np

A = np.arange(14, 2, -1).reshape(3, 4)

print(A)

# 求矩阵的索引

print(np.argmin(A))  # 求最小的索引
print(np.argmax(A))  # 求最大的索引

# 求矩阵的平均值
print(np.mean(A, axis=0))  # 0列1行
print(np.mean(A, axis=1))  # 0列1行
print(np.average(A))

# 求矩阵的中位数

print(np.median(A))

# 和
print(np.cumsum(A))

# 差
print(np.diff(A))

# 排序

print(np.sort(A))  # 这儿表示每一行来进行排序

# 矩阵的转置

print(A.T)

# 矩阵的过滤
print(np.clip(A, 5, 9))  # 这儿解释一下表示 在=在这个矩阵中过滤元素为5到9之间,如果小于5就等于5,如果大于9等于9

# 矩阵元素的打印

print(A[:, 1])  # 打印第二列的元素
print(A[1, 1:2])  # 打印第二行中2到3的元素 取不到尾部
