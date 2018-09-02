import pandas as pd
import numpy as np

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])

# 数据的筛选


print(df)
print(df['A'])
print(df.A)
print("=" * 10)
print(df[0:3])

print(df['20130101':'20130103'])
# 纯标签筛选
print("=" * 10)
print(df.loc['20130102'])

print("=" * 10)
print(df.loc['20130102', ['A', 'B']])
# 纯数字帅选
print("=" * 10)
print(df.iloc[3:5, 1:3])

print("=" * 10)
print(df.iloc[[1, 3, 5], 1:3])

# 数字+标签一起筛选
print("=" * 10)
print(df.ix[:3, ['A', 'B']])

# boolean帅选
print("=" * 10)
print(df[df.A > 8])
