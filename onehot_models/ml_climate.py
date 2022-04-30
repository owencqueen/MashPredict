import numpy as np

import sys; sys.path.append('../utils')
from extract import *
from onehot import *

from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

#meta = get_loc_metadata()
meta = get_climate()
#mat, y, mapper = get_onehot(meta, path = '/data1/compbio/oqueen/poplar/MashPredict/poplar_onehot.txt')
mat, y, mapper = get_onehot(meta, yname = 'Mean_Temp')


# def run_model():

#     #model = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5, multi_class = 'ovr')

#     score = cross_val_score(model, mat, y, verbose = 2, n_jobs = 3)

#     print('Score:', score)

def run_pca_model():

    pca = PCA(n_components = 100)
    mat_reduce = pca.fit_transform(mat)

    mat_reduce, newy = shuffle(mat_reduce, y)

    #model = LogisticRegression()
    model = RandomForestRegressor()

    score = cross_val_score(model, mat_reduce, newy, verbose = 2, n_jobs = 3)

    print('Score:', score)
    print('Avg. score', np.mean(score))

if __name__ == '__main__':
    run_pca_model()