import numpy as np

data = np.random.normal(3., 0.5)
a = [1, 2, 2, 3]
b = [[1, 3, 4, 5, 6], [1, 3, 4, 5, 7], [1, 3, 4, 5, 8], [1, 3, 4, 5, 9], [1, 3, 4, 5, 10]]
mat = np.array(b)
c = mat[:, :3]
print(c)
for i in range(1, 4):
    print(sum(a[:i]))
    # print(a[1:])
# print(data)
