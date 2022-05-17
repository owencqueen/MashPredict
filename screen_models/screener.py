import numpy as np
import pandas as pd
from functools import partial

import sys; sys.path.append('../utils')
from extract import *
from onehot import *

from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import GradientBoostingRegressor as XGB
from sklearn.linear_model import Lasso as Lasso
from sklearn.linear_model import ElasticNet as ElasticNet
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV, cross_val_score


meta = get_climate()
# mat, y, mapper = get_onehot(meta, yname = 'Mean_Temp', \
#     path = '/data1/compbio/oqueen/poplar/MashPredict/poplar_onehot.txt')

get_OH = partial(get_onehot, 
    meta = meta,
    path = '/data1/compbio/oqueen/poplar/MashPredict/poplar_onehot.txt')

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

# Screen over models

# All possible models
targets = list(meta.columns)
targets.remove('Geno')

#score_mat = np.zeros((len(targets), len(models_dict.keys())))

def gsearch_screen():
    df = pd.DataFrame(index = targets, columns = list(models_dict.keys()))

    for targ in targets:
        OH = get_OH(yname = targ)
        for m in models_dict.keys():
            score, params = fit_eval_model(*OH, m)
            print(f'Model : {m}, {params} \t Target: {targ} \t Score: {score}')
            df.loc[targ, m] = score

    return df

def plain_screen():
    df = pd.DataFrame(index = targets, columns = list(models_dict.keys()))

    for targ in targets:
        OH = get_OH(yname = targ)
        X = PCA(n_components=50).fit_transform(OH[0])
        for m in models_dict.keys():
            score = fit_eval_model_CV(X, OH[1], m)
            print(f'Model : {m} \t Target: {targ} \t Score: {score}')
            df.loc[targ, m] = score

    return df

if __name__ == '__main__':
    df = plain_screen()
    df.to_csv('screen_scores.csv') 