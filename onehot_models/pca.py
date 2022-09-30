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

pca = PCA()
pca.fit(mat)

evar = sorted(pca.explained_variance_)

cum_evar = np.cumsum(evar)
plt.plot(evar)
plt.ylabel('Pct. Explained Variance')
plt.xlabel('Components')
plt.title('Mash Sketch PCA - Poplar')
plt.show()
