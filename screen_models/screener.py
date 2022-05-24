import argparse
import numpy as np
import pandas as pd
from functools import partial

import sys; sys.path.append('../utils')
from extract import *
from onehot import *
from sample import EPS_sample_mask

from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import GradientBoostingRegressor as XGB
from sklearn.linear_model import Lasso as Lasso
from sklearn.linear_model import ElasticNet as ElasticNet
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

models_dict = {
    'ENet': ElasticNet,
    'RF': RF,
    'Lasso': Lasso,
    'XGB': XGB
}

# Hyperparameters:
enet_params = {
    'alpha': [0.5, 1.0, 1.5],
    'l1_ratio': [0, 0.25, 0.5, 0.75]
}

rf_params = {
    'n_estimators': [50, 100, 150]
}

xgb_params = {
    'learning_rate': [0.05, 0.1, 0.5],
    'n_estimators': [50, 100, 150]
}

lasso_params = {
    'alpha': [0.01, 0.5, 1.0, 1.5]
}

params_dict = {
    'ENet': enet_params,
    'RF': rf_params,
    'Lasso': lasso_params,
    'XGB': xgb_params
}

def OH():
    meta = get_full_meta()

    get_OH = partial(get_onehot, 
        meta = meta,
        path = '/data1/compbio/oqueen/poplar/MashPredict/poplar_onehot.txt')

    targets = list(meta.columns)

    return get_OH, targets

def get_metadata(filter_columbia = False):
    #meta = get_loc_metadata()
    #y = meta['Full_class']
    meta = get_full_meta()
    targets = list(meta.columns)
    print(targets)
    return meta, targets

def get_dist_y(meta, target, EPS_bounds = None):
    # Gets the actual distance matrix and filters EPS if needed

    dist, ymask = make_distance_matrix(meta.index.tolist())
    meta_masked = meta.loc[ymask,:]
    y = meta_masked.loc[:,target]

    if EPS_bounds is not None:
        print(EPS_bounds)
        mask, _, _ = EPS_sample_mask(meta_masked, target, float(EPS_bounds[0]), float(EPS_bounds[1]))
        y = y.loc[mask]
        dist = dist[np.nonzero(mask)[0],:][:,np.nonzero(mask)[0]]

    return dist, y

def get_OH_y(meta, target, EPS_bounds = None):
    get_OH = partial(get_onehot, 
        meta = meta,
        path = '/data1/compbio/oqueen/poplar/MashPredict/poplar_onehot.txt')

    X, y, mapper = get_OH(yname = target)

    if EPS_bounds is not None:
        pass

    return X, y

def fit_eval_model_gsearch(mat, y, mapper, mname):
    clf = GridSearchCV(models_dict[mname](), 
        params_dict[mname], scoring = 'r2',
        n_jobs = 5, verbose = 3)
    clf.fit(mat, y)
    score = clf.best_score_
    params = clf.best_params_
    return score, params

def fit_eval_model_CV(mat, y, mname):
    est = models_dict[mname]()
    score = cross_val_score(est, X = mat, y = y, 
        scoring = 'r2', n_jobs = 5,
        verbose = 3)
    return np.mean(score)

def gsearch_screen_OH():
    df = pd.DataFrame(index = targets, columns = list(models_dict.keys()))

    get_OH, targets = OH() # Get partial function

    for targ in targets:
        OH = get_OH(yname = targ)
        for m in models_dict.keys():
            score, params = fit_eval_model(*OH, m)
            print(f'Model : {m}, {params} \t Target: {targ} \t Score: {score}')
            df.loc[targ, m] = score

    return df

def plain_screen_OH(EPS_bounds = None):
    meta, targets = get_metadata()
    df = pd.DataFrame(index = targets, columns = list(models_dict.keys()))

    for targ in targets:
        X, y = get_OH_y(meta, targ, EPS_bounds = EPS_bounds)
        X = PCA(n_components=50).fit_transform(X)
        for m in models_dict.keys():
            score = fit_eval_model_CV(X, y, m)
            print(f'Model : {m} \t Target: {targ} \t Score: {score}')
            df.loc[targ, m] = score

    return df

def val_over_k(dist, y, regression = False):
    # Validates over k values for Knn
    scores = []
    max_score = -1e5
    max_k = -1

    for k in range(1, 20):

        if regression:
            est = KNeighborsRegressor(n_neighbors=k,
                metric = 'precomputed')

            score = cross_val_score(est, 
                X = dist,
                y = y,
                scoring = 'r2',
                verbose = 0
                )

        else:
            est = KNeighborsClassifier(n_neighbors=k,
                metric = 'precomputed')

            score = cross_val_score(est, 
                X = dist,
                y = LabelEncoder().fit_transform(y),
                scoring = 'accuracy',
                verbose = 0
                )

        if max_score < np.mean(score):
            max_score = np.mean(score)
            max_k = k

    return max_score, max_k

def screen_dist(EPS_bounds = None):
    # Run the screen over distance-based
    meta, targets = get_metadata()
    df = pd.DataFrame(index = targets, columns = ['k', 'score'])

    for targ in targets:

        dist, y = get_dist_y(meta, targ, EPS_bounds = EPS_bounds)

        score, k = val_over_k(dist, y, 
            regression = (targ != 'Full_class'))

        print(f'K : {k} \t Target: {targ} \t Score: {score}')
        df.loc[targ, 'k'] = k
        df.loc[targ, 'score'] = score

    return df

if __name__ == '__main__':

    #OH()

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required = True)

    group.add_argument('--OH', action = 'store_true', help = 'Run one-hot')
    group.add_argument('--dist', action = 'store_true', help = 'Run distance')
    parser.add_argument('--gsearch', action = 'store_true', help = 'Runs grid search')
    parser.add_argument('--EPS', nargs = 2, default = None, help = 'Bounds in [0,1] for EPS, in order. Ex: "--EPS 0.25 0.75"')
    parser.add_argument('--target_file', type = str, help = 'Output file name for screening results')
    parser.add_argument('--exclude_columbia', action = 'store_true', help = 'If included, exclude Columbia group')

    args = parser.parse_args()

    if (args.dist):
        print('Running screen over k values even if args.gsearch is false')

    if args.OH and (not args.dist): # Default to below if args.dist provided
        pass
    else: # Assume dist if OH not provided
        df = screen_dist(args.EPS)
        df.to_csv(args.target_file)



    # df = plain_screen()
    # df.to_csv('screen_scores.csv') 