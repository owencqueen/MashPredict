import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np


df = pd.read_csv('test.csv', index_col=0)
df = df.iloc[:,1:]

avg = [-1.0 * np.mean(df.iloc[i,:]) for i in range(df.shape[0])]
df = df.iloc[np.argsort(avg),:] # Sort by average score across all models (largest is first)

sns.heatmap(df, cmap = 'viridis')
plt.show()

# Make barplot:
xbase = np.arange(df.shape[0])

for i in range(df.shape[1]):
    plt.bar(xbase + 0.15 * (i + 1), df.iloc[:,i], width = 0.15, align = 'edge', label = df.columns[i])

print(df.index)
plt.xticks((xbase + 0.15 * (df.shape[0] / 2)) - 1, labels = list(df.index), rotation = 'vertical')
plt.ylim([-1.5, np.max(df.to_numpy()) * 1.05])
plt.legend()
plt.tight_layout()
plt.show()
