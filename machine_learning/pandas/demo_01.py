import numpy as np
import pandas as pd

s = pd.Series([1, 3, 6, np.nan, 44, 1])
print(s)
dates = pd.date_range('20160101', periods=6)
print(dates)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
print(df)

df1 = pd.DataFrame(np.arange(12).reshape((3, 4)))
print(df1)
print(df1.dtypes)
print(df1.values)
print(df1.columns)
print(df1.describe())

print(df.T)

# 排序
sort = df.sort_index(axis=0, ascending=False)
print(sort)

sort1 = df.sort_values(by='a')
print(sort1)