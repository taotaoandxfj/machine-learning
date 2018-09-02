import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Series

data = pd.Series(np.random.randn(1000), index=np.arange(1000))
print(data)
data = data.cumsum()

# DataFrame

data = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000),
                    columns=list("ABCD"))
data = data.cumsum()
print(data.head(5))

# plot methods
# 'bar','hist','box','area','scatter','hexbin','pie'


# data.plot()
ax = data.plot.scatter(x='A', y='B', color='DarkBlue', label='Classs 1')
# data.plot.scatter(x='A', y='C', color='DarkGreen', label='Classs 2', ax=ax)

plt.show()
