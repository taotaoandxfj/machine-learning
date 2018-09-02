import pandas as pd
import numpy as np

ratings_df = pd.read_csv('ratingsProccessed.csv')

for index, row in ratings_df.iterrows():
    print(index)  # 表示序号
    print(row)  # 表示一行

ratings_df_length = np.shape(ratings_df)[0]
print(ratings_df_length)