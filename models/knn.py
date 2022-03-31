import sys; sys.path.append('/data1/compbio/oqueen/poplar/MashPredict/utils')
import numpy as np
import pandas as pd

from extract import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def cross_val_classify():

    meta = get_loc_metadata()
    y = meta['Full_class']
    ykey = meta['Geno'].tolist()
    dist, ymask = make_distance_matrix(ykey)

    y = y.iloc[list(np.nonzero(ymask)[0])].tolist()

    scores = []

    for k in range(20):
        est = KNeighborsClassifier(n_neighbors=k,
            metric = 'precomputed')

        score = cross_val_score(est, 
            X = dist,
            y = y,
            scoring = 'accuracy',
            verbose = 0
            )
    
        print('{} neighbors = {}'.format(k, np.mean(score)))

        scores.append(score)

    # plt.plot(score)
    # plt.title('kNN')
    # plt.xlabel('k')
    # plt.ylabel('Accuracy')
    # plt.show()


if __name__ == '__main__':
    cross_val_classify()
