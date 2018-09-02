import numpy as np

array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int)
print(array)  # 打印矩阵
print("number of dim:", array.ndim)  # 表示该矩阵是几维的 2
print("shape:", array.shape)  # (2, 3)
print("size:", array.size)  # 6
print("dtype:", array.dtype)

array1 = np.zeros((3, 4), dtype=int)
print(array1)

array2 = np.ones((3, 4))
print(array2)

a = np.arange(10, 21, 2).reshape(2, 3)

print(a)

b = np.linspace(1, 10, 5).reshape(5, 1)  # (10-1)/(5-1)
print(b)
