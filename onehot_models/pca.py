import numpy as np

import sys; sys.path.append('../utils')
from extract import *
from onehot import *

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

meta = get_loc_metadata()
mat, y, mapper = get_onehot(meta)

le = LabelEncoder()
yord = le.fit_transform(y)

pca = PCA(n_components = 2)
pca = fit(mat)

evar = np.cumsum(pca.explained_variance_)
plt.plot(evar)
plt.show()