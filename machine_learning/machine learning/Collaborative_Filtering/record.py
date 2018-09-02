import numpy as np

a = [[1, 3, 0, 3, 4, 0], [2, 3, 0, 3, 4, 0]]
a = np.array(a)
idx = a[0, :] != 0
rating_mean = np.mean(a[0, idx])
print(a[0, idx])
