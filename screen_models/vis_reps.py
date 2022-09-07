import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

# Build df, use code below
# Feeds in the csv files
onehot = pd.read_csv('s4000_OH_res.csv', index_col=0)
dist = pd.read_csv('dist_res.csv', index_col = 0)
aligned = pd.read_csv('aligned_res.csv', index_col = 0)

def get_max_scores(df):
    arr = df.to_numpy()
    arr = np.nan_to_num(arr)
    print(arr)
    print('max', arr.max(axis=1))
    return arr.max(axis=1)

df = pd.DataFrame(
    {
        #Uncomment for distance 'Pairwise Dist.': dist.loc[:,'score'],
        'Alignment-based': get_max_scores(aligned),
        'Sketch-based': get_max_scores(onehot),
        'Distance-based': dist.loc[:,'score']
    },
    index = onehot.index)

df = df.sort_values(by = 'Alignment-based', ascending=False)

# Make barplot:
xbase = np.arange(df.shape[0])

for i in range(df.shape[1]):
    nan_mask = df.iloc[:,i].isna()
    df.iloc[:,i].loc[nan_mask] = 0
    plt.bar(xbase + 0.15 * (i + 1), df.iloc[:,i], width = 0.15, align = 'edge', label = df.columns[i])

plt.xticks((xbase + 0.15 * (df.shape[0] / 2)) - 1, labels = list(df.index), rotation = 'vertical')
plt.ylim([-1.5, np.max(df.to_numpy()) * 1.05])
plt.legend()
plt.ylabel('$R^2$')
plt.tight_layout()
plt.show()
