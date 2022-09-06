import sys
sys.path.append('/data1/compbio/oqueen/poplar/MashPredict/utils')
sys.path.append('/lustre/isaac/scratch/oqueen/MashPredict/utils')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from extract import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import plot_confusion_matrix, confusion_matrix

def compute_distance_error(gt, pred):

    mean_loss = np.mean(np.sqrt(np.sum((((pred - gt) * 111.139)) ** 2, dim = 0)))
    mean_lat_dist = np.mean((np.abs(pred - gt)[0,:] * 111.139))
    mean_lon_dist = np.mean((np.abs(pred - gt)[1,:] * 111.139))

    return mean_loss, mean_lat_dist, mean_lon_dist

def knn_CM(random_state = None):

    #meta = get_loc_metadata()
    meta = get_full_meta()
    # ylat = meta['Latitude']
    # ylon = meta['Longitude']
    ykey = meta.index.tolist()
    dist, ymask = make_distance_matrix(ykey)

    ylat = meta.loc[np.nonzero(ykey)[0], 'Latitude'].to_numpy()
    ylon = meta.loc[np.nonzero(ykey)[0], 'Longitude'].to_numpy()

    y = np.stack([ylat, ylon]).T # Shape = (N,2)

    clf = MultiOutputClassifier(KNeighborsRegression(n_neighbors=4, metric = 'precomputed'))

    preds = cross_val_predict(clf, X = dist, y = y)

    mean_loss, mean_lat_dist, mean_lon_dist = compute_distance_error(y, preds)

    return (mean_loss, mean_lat_dist, mean_lon_dist), (preds, gt)

if __name__ == '__main__':
    knn_CM()