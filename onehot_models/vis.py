import numpy as np

import sys; sys.path.append('../utils')
from extract import *
from onehot import *

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams, cycler

meta = get_loc_metadata()
mat, y, mapper = get_onehot(meta)

le = LabelEncoder()
yord = le.fit_transform(y)

exclude_yun = []
cmap = plt.cm.viridis
rcParams['axes.prop_cycle'] = cycler(color=cmap(list(range(5 - len(exclude_yun)))))
cust = [
    Line2D([0], [0], marker = 'o', color = cmap(i/(4 - len(exclude_yun))), label = le.classes_[i] , markersize = 10) \
        for i in range(5 - len(exclude_yun))
]

#pca_X = PCA(n_components = 2).fit_transform(mat)
reduced_X = Isomap(n_components=2).fit_transform(mat)

plt.scatter(reduced_X[:,0], reduced_X[:,1], c = yord)
plt.ylabel('Isomap 2')
plt.xlabel('Isomap 1')
plt.legend(handles = cust)
plt.show()
