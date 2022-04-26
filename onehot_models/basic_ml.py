import numpy as np

import sys; sys.path.append('../utils')
from extract import *
from onehot import *

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_score

meta = get_loc_metadata()
mat, y, mapper = get_onehot(meta, path = '/data1/compbio/oqueen/poplar/MashPredict/poplar_onehot.txt')

def run_model():

    model = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5, multi_class = 'ovr')

    score = cross_val_score(model, mat, y, verbose = 2, n_jobs = 3)

    print('Score:', score)

if __name__ == '__main__':
    run_model()
