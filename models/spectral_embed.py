import sys; sys.path.append('../utils')
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from extract import *

from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import kneighbors_graph as knn_graph
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC

def spectral_embed():
    meta = get_loc_metadata()
    y = meta['Full_class']
    ykey = meta['Geno'].tolist()
    dist, ymask = make_distance_matrix(ykey)

    y = y.iloc[list(np.nonzero(ymask)[0])].tolist()

    print(y)

    yordinal = LabelEncoder().fit_transform(y)
    print(yordinal[:20])

    spec = SpectralEmbedding(n_components = 8, affinity = 'precomputed')

    xembed = spec.fit_transform(dist)

    # Run SVM:
    svm = RFC()

    score = cross_val_score(svm, 
        X = xembed,
        y = yordinal,
        scoring = 'accuracy',
        verbose = 0,
        cv = 5
        )

    print(score)


    # plt.scatter(xembed[:,0], xembed[:,1], c = yordinal)
    # plt.show()

if __name__ == '__main__':
    spectral_embed()