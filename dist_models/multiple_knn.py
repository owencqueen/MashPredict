import sys, argparse
sys.path.append('/data1/compbio/oqueen/poplar/MashPredict/utils')
sys.path.append('/lustre/isaac/scratch/oqueen/MashPredict/utils')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from extract import *

from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split
from sklearn.metrics import plot_confusion_matrix, confusion_matrix

def compute_distance_error(gt, pred):

    mean_loss = np.mean(np.sqrt(np.sum((((pred - gt) * 111.139)) ** 2, axis = 0)))
    mean_lat_dist = np.mean((np.abs(pred - gt)[0,:] * 111.139))
    mean_lon_dist = np.mean((np.abs(pred - gt)[1,:] * 111.139))

    return mean_loss, mean_lat_dist, mean_lon_dist

def knn_CM(mashsize, random_state = None):

    #meta = get_loc_metadata()
    meta = pd.read_csv('/lustre/isaac/scratch/oqueen/MashPredict/all_meta.csv', sep = '\t', index_col = 0)
    dist_path = '/lustre/isaac/scratch/oqueen/MashPredict/s{}_dist.txt'.format(mashsize)
    # ylat = meta['Latitude']
    # ylon = meta['Longitude']
    ykey = meta.index.tolist()
    dist, ymask = make_distance_matrix(ykey, data_path = dist_path)

    ylat = meta.iloc[np.nonzero(ymask)[0]]['Latitude'].to_numpy()
    ylon = meta.iloc[np.nonzero(ymask)[0]]['Longitude'].to_numpy()

    y = np.stack([ylat, ylon]).T # Shape = (N,2)

    # Manual cross validation setup:
    #

    # inds = np.arange(y.shape[0])
    # preds = np.zeros_like(y)
    # for train_idx, test_idx in kf.split(inds):
    #     clf = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=4, metric = 'precomputed'))
    #     train_mat, ytrain = dist[train_idx,:][:,train_idx], y[train_idx,:]
    #     test_mat, ytest = dist[test_idx,:][:,test_idx], y[test_idx,:]

    #     print('size train_mat', train_mat.shape)
    #     print('size test_mat', test_mat.shape)

    #     clf.fit(train_mat, ytrain)
    #     preds[test_idx,:] = clf.predict(test_mat)

    kf = KFold(n_splits = 5, shuffle = True, random_state = random_state)
    inds = np.arange(y.shape[0])
    preds = np.zeros_like(y)
    scores, lat_scores, lon_scores = [], [], []

    for train_idx, test_idx in kf.split(inds):
        trainset, ytrain = dist[train_idx,:][:,train_idx], y[train_idx,:]
        test_mat, ytest = dist[test_idx,:][:,test_idx], y[test_idx,:]

        # Split validation set:
        train_idx, val_idx = train_test_split(np.arange(trainset.shape[0]), test_size = 0.1, shuffle = True)
        train_mat, ytrain = trainset[train_idx,:][:,train_idx], y[train_idx,:]
        val_mat, yval = trainset[val_idx,:][:,val_idx], y[val_idx,:]

        best_error = 1e9
        best_clf = None
        best_k = 0

        for k in range(1, 31):
            clf = KNeighborsRegressor(n_neighbors=k, metric = 'precomputed')
            clf.fit(train_mat, ytrain)
            preds_val = clf.predict(val_mat)

            mean_loss, _, _ = compute_distance_error(yval, preds_val)

            if mean_loss < best_error:
                best_error = mean_loss
                best_clf = clf
                best_k = k

        preds[test_idx] = best_clf.predict(test_mat)

        mean_loss, lat_loss, lon_loss = compute_distance_error(ytest, preds[test_idx])
        print('(k = {:02d}) Mean dist error: {:.4f}'.format(k, mean_loss))

        scores.append(mean_loss); lat_scores.append(lat_loss); lon_scores.append(lon_loss) 

    return (scores, lat_scores, lon_scores), (preds, y)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mashsize', type = str, default = '4k')
    args = parser.parse_args()

    map_to_mashsize = {
        '500': '500',
        '2k': '2000',
        '4k': '4000',
        '50k': '50000'
    }

    knn_CM(mashsize = map_to_mashsize[args.mashsize])