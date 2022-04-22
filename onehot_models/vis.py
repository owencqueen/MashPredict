import numpy as np

import sys; sys.path.append('../utils')
from extract import *
from onehot import *

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

meta = get_loc_metadata()
mat, y, mapper = get_onehot(meta)

le = LabelEncoder()
yord = le.fit_transform(y)

#pca_X = PCA(n_components = 2).fit_transform(mat)
reduced_X = Isomap(n_components=2).fit_transform(mat)

plt.scatter(reduced_X[:,0], reduced_X[:,1], c = yord)
plt.ylabel('Isomap 2')
plt.xlabel('Isomap 1')
plt.show()
