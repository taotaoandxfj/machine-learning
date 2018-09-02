import pandas as pd
import numpy as np

# concatenating

df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
df3 = pd.DataFrame(np.ones((3, 4)) * 2, columns=['a', 'b', 'c', 'd'])

# 合并df1 df2 df3

res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)  # 上下合并

print(res)

# join{'inner','outer'}

df4 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
df5 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])

print(df4)
print("=" * 10)
print(df5)

print("=" * 10)

res1 = pd.concat([df4, df5], join='outer')
res2 = pd.concat([df4, df5], join='inner', ignore_index=True)
res3 = pd.concat([df4, df5], axis=1, join_axes=[df4.index])
s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])

# append

res4 = df4.append([df1, df1], ignore_index=True)  # 竖向添加
print(res1)
print("=" * 10)
print(res2)
print("=" * 10)
print(res3)
print("=" * 10)
print(res4)
