import sys; sys.path.append('/data1/compbio/oqueen/poplar/MashPredict/utils')
sys.path.append('../utils')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from extract import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import plot_confusion_matrix, confusion_matrix

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

def knn_CM():

    meta = get_loc_metadata()
    y = meta['Full_class']
    ykey = meta['Geno'].tolist()
    dist, ymask = make_distance_matrix(ykey)

    y = y.iloc[list(np.nonzero(ymask)[0])]

    print(y.value_counts())
    
    y = y.tolist()

    yun_sorted = sorted(np.unique(y))

    est = KNeighborsClassifier(n_neighbors=4,
            metric = 'precomputed')

    preds = cross_val_predict(est, X = dist, y = y)

    # Show CM:

    CM = confusion_matrix(y, preds)

    dict_for_pd = {yun_sorted[i]:CM[i] for i in range(len(yun_sorted))}

    df = pd.DataFrame(dict_for_pd, index = yun_sorted)

    sns.heatmap(df, annot = True, cmap = plt.cm.viridis, fmt = 'd', yticklabels = True)
    plt.yticks(rotation=0)
    plt.show()


if __name__ == '__main__':
    #cross_val_classify()
    knn_CM()
