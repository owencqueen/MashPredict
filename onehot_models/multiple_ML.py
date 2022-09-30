import sys, argparse, pickle, os, warnings
sys.path.append('/data1/compbio/oqueen/poplar/MashPredict/utils')
sys.path.append('/lustre/isaac/scratch/oqueen/MashPredict/utils')

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

# import warnings
# warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

from extract import *

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from geopy.distance import geodesic

# from sklearn.utils.testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning

import itertools

def get_random_state(i):
    # Deterministic method to compute random state for model
    return i * 587 % 103 + 127

# def compute_distance_error(gt, pred):

#     mean_loss = np.mean(np.sqrt(np.sum((((pred - gt) * 111.139)) ** 2, axis = 0)))
#     mean_lat_dist = np.mean((np.abs(pred - gt)[0,:] * 111.139))
#     mean_lon_dist = np.mean((np.abs(pred - gt)[1,:] * 111.139))

#     return mean_loss, mean_lat_dist, mean_lon_dist

def compute_distance_error(gt, pred):
    #print(gt.shape)
    mean_loss = np.mean(np.sqrt([geodesic(gt[i,:], pred[i,:]).km ** 2 for i in range(gt.shape[0])]))
    return mean_loss

def prep(df, meta, geno_axis = 1, filter_X = True):
    
    # Sort based on sample
    # Take union of both

    if geno_axis == 1:
        all_genos = list(set(df.columns.to_list()).intersection(set(meta.index.to_list())))
        X = np.stack([df.loc[:,n].to_numpy() for n in all_genos])
    elif geno_axis == 0:
        all_genos = list(set(df.index.to_list()).intersection(set(meta.index.to_list())))
        X = np.stack([df.loc[n,:].to_numpy() for n in all_genos])

    # Filter X's by low frequency
    if filter_X:
        exclude = [1, 2, 3, X.shape[0] - 1, X.shape[0]]
        mask = np.ones((X.shape[1],), dtype = bool)
        sum_mat = X.sum(axis=0)
        for i in exclude:
            mask &= (sum_mat != i)

        X = X[:,mask]
    #meta_y = np.stack([meta.loc[n,:] for n in all_genos])
    meta_y = meta.loc[all_genos,:]

    return X, meta_y

def prepare_data(txt_path = '../data/trimmed_50000.txt', geno_axis = 1, latlong = True):
    #df = pd.read_csv('../onehot_s50000.txt', sep='\t', index_col=0)
    df = pd.read_csv(txt_path, sep='\t', index_col=0)
    if latlong:
        meta = pd.read_csv('../all_meta.csv', sep='\t', index_col=0)
    else:
        meta = pd.read_csv('../poplar_river_trim.csv', index_col = 0)
    filter_X = (txt_path != '../data/trimmed_50000.txt') and ('pca' not in txt_path)
        
    return prep(df, meta, filter_X = filter_X, geno_axis = geno_axis)

#@ignore_warnings(category=ConvergenceWarning)
def runmodel(X, meta_y, modelname, random_state = None, latlong = True):

    if modelname == 'elasticnet':
        search_params = {
            'l1_ratio': [0.01],
            'alpha': [0.1]
        }
    
    elif modelname == 'logistic':
        search_params = {
            'penalty': ['l2'],
            'C': [1.0]
        }

    else:
        search_params = {
            'learning_rate': [0.1],
            'n_estimators': [150]
        }

    if latlong:
        y = np.stack([meta_y['Latitude'].to_numpy(), meta_y['Longitude'].to_numpy()]).T
    else:
        y = LabelEncoder().fit_transform(meta_y['River'].to_numpy())

    # Fit two separate models:
    k = list(search_params.keys())
    v = list(search_params.values())

    best_model = None
    best_error = 1e9

    scores = []

    for vals in itertools.product(*v):
        match_kv = {k[i]:vals[i] for i in range(len(k))}

        if latlong:
            if modelname == 'elasticnet':
                model = ElasticNet(**match_kv)

            elif modelname == 'xgboost':
                model = GradientBoostingRegressor(**match_kv)

            regr = MultiOutputRegressor(model, n_jobs = 2)

            #regr = make_pipeline(StandardScaler(), regr)
            preds = cross_val_predict(regr, X, y)
            # preds = []
            # kf = KFold(n_splits = 5, shuffle = False)
            # for train_inds, test_inds in kf.split(X):
            #     print('train', train_inds)
            #     Xtrain, ytrain, Xtest, ytest = X[train_inds], y[train_inds], X[test_inds], y[test_inds]
            #     ss = StandardScaler().fit(ytrain) # Fit SS on ytrain
            #     ytrain = ss.transform(ytrain)
            #     regr.fit(Xtrain, ytrain)
            #     pred_raw = regr.predict(Xtest)
            #     preds.append(ss.inverse_transform(pred_raw))
            # # print('mu, std lat', np.mean(preds[:,0]), np.std(preds[:,0]))
            # # print('mu, std long', np.mean(preds[:,1]), np.std(preds[:,1]))
            # preds = np.concatenate(preds, axis = 0)
            # print('preds', preds.shape)
            error = compute_distance_error(y, preds)

        else:
            if modelname == 'logistic':
                model = LogisticRegression(**match_kv)

            elif modelname == 'xgboost':
                model = GradientBoostingClassifier(**match_kv)

            # This error is actually a score (f1 score)
            #model = make_pipeline(StandardScaler(), model)
            error = cross_val_score(model, X, y, scoring = 'f1_macro')
            
        #print('error', error)
        scores.append(error)

    return scores

def run_all_models(X, meta_y, modelname, latlong = True):

    all_scores = []
    for i in trange(30):
        Xshuf, meta_shuf = shuffle(X, meta_y, random_state = get_random_state(i))
        scores = runmodel(Xshuf, meta_shuf, modelname = modelname, random_state = get_random_state(i), latlong = latlong)
        all_scores.append(np.mean(scores))

    print('Dist: {:.4f} +- {:.4f}'.format(np.mean(all_scores), np.std(all_scores) / np.sqrt(50)))

def run_all_models_cvsplit(X, meta_y, modelname, start, stop, scorepath, latlong = True):

    all_scores = []
    range_nums = np.arange(start, stop)
    for i in tqdm(range_nums):
        Xshuf, meta_shuf = shuffle(X, meta_y, random_state = get_random_state(i))
        scores = runmodel(Xshuf, meta_shuf, modelname = modelname, random_state = get_random_state(i), latlong = latlong)
        all_scores.append(np.mean(scores))

    pickle.dump(all_scores, open(scorepath, 'wb'))

    print('Dist: {:.4f} +- {:.4f}'.format(np.mean(all_scores), np.std(all_scores) / np.sqrt(50)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mashsize', default = '500', type = str)
    parser.add_argument('--model', default = 'elasticnet', type = str)
    parser.add_argument('--scorepath', type = str, default = 'test', 
        help = "File to which to save to. Don't include suffix of filename.")
    parser.add_argument('--start', type = int, default = None)
    parser.add_argument('--stop', type = int, default = 30)
    parser.add_argument('--river', action = 'store_true')
    parser.add_argument('--aligned', action = 'store_true')
    args = parser.parse_args()

    fname_map = {
        '500': '../data/onehot_s500.txt',
        '2k' : '../data/onehot_s2000.txt',
        '4k' : '../data/onehot_s4000.txt',
        '50k': '../data/trimmed_50000.txt'
    }

    if args.aligned:
        fname = '../1323_pca.txt'
        spath = '{}_aligned.pickle'.format(args.scorepath)
        geno_axis = 0
    else:
        fname = fname_map[args.mashsize]
        spath = '{}_{}-{}.pickle'.format(args.scorepath, args.start, args.stop)
        geno_axis = 1
        
    X, meta_y = prepare_data(fname, geno_axis = geno_axis, latlong = (not args.river))

    if args.start is not None:
        run_all_models_cvsplit(X, meta_y, modelname = args.model, 
            start = args.start, stop = args.stop, scorepath = spath, latlong = (not args.river))
    else:
        run_all_models(X, meta_y, args.model, latlong = (not args.river))