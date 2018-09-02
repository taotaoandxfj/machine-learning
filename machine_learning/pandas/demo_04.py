import pandas as pd
import numpy as np

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
df.iloc[0, 1] = np.nan
df.iloc[1, 2] = np.nan
print(df)

print("=" * 10)
print(df.dropna(axis=0, how='any'))  # how={"any","all"} any表示有一个nan就丢掉,all表示有一个就丢掉 0表示保留列(丢掉行) 1表示保留行

print(df.fillna(value=0))  # 将空的数据填为0

# 判断数据表中是否有数据的缺失
print(df.isnull())  # 会返回true or false

# 如果表中数据不好看的话 可以用一下的方式来进行判断
print(np.any(df.isnull()))
