import pandas as pd
import numpy as np

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']
                     })
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']
                      })

print(left)
print(right)
print("=" * 10)
res = pd.merge(left, right, on='key')
print(res)

# consider two

left2 = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                      'key2': ['K0', 'K1', 'K0', 'K1'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']
                      })
right2 = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                       'key2': ['K0', 'K0', 'K0', 'K0'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']
                       })

print(left2)
print(right2)

res2 = pd.merge(left2, right2, on=['key1', 'key2'], how="outer")  # 默认的方法是inner
res3 = pd.merge(left2, right2, on=['key1', 'key2'], how="right")  # 默认的方法是inner
print(res2)

print("=" * 10)
print(res3)

df1 = pd.DataFrame({'col1': [0, 1], 'col_left': ['a', 'b']})
df2 = pd.DataFrame({'col1': [1, 2, 2], 'col_right': [2, 2, 2]})
print(df1)
print(df2)

res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
res4 = pd.merge(df1, df2, on='col1', how='outer', indicator="indicator_column")

print(res4)

left3 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                      'B': ['B0', 'B1', 'B2']

                      }, index=['K0', 'K1', 'K2'])

right3 = pd.DataFrame({'C': ['C0', 'C1', 'C2'],
                       'D': ['D0', 'D1', 'D2']

                       }, index=['K0', 'K2', 'K3'])

res5 = pd.merge(left, right, left_index=True, right_index=True, how='outer')
# res5 = pd.merge(left, right, left_index=True, right_index=True, how='inner')

print(res5)
