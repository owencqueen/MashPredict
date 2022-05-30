from tkinter import Label
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

def process():
    df = pd.read_csv('../poplar_onehot.txt', sep = '\t')
    classes = pd.read_csv('../all_classes.txt', sep = '\t')
    #y = classes['Full_class'].to_numpy()

    mask = np.array([('GW' in df.columns[i]) for i in range(df.shape[1])])
    test_cols = df.columns[mask]

    Xtest = df.loc[:,test_cols].to_numpy()
    #ytest = classes['Full_class'].loc[mask].to_numpy()

    df_train = df.loc[:, ~mask]
    inter = list(set(df_train.columns).intersection(classes['Geno']))
    Xtrain = df_train.loc[:,inter].to_numpy()

    cols_train = list(classes['Geno'])

    reorder = [cols_train.index(s) for s in inter]

    #.to_numpy()
    
    ytrain = classes['Full_class'].to_numpy()
    ytrain = ytrain[reorder]

    # Split out

    return np.transpose(Xtrain), np.transpose(Xtest), ytrain, df.columns[mask]

def predict():

    Xtrain, Xtest, ytrain, GWsamples = process()
    print(Xtrain.shape)
    print(ytrain.shape)
    print(Xtest.shape)

    LE = LabelEncoder()
    ytrain = LE.fit_transform(ytrain)

    pca = PCA(n_components = 50)
    pca.fit(Xtrain)

    Xtrain_pca = pca.transform(Xtrain)
    Xtest_pca = pca.transform(Xtest)

    model = GradientBoostingClassifier()

    model.fit(Xtrain_pca, ytrain)

    yhat = model.predict(Xtest_pca)
    print(yhat)
    #scores = cross_val_score(model, X, y, scoring = 'accuracy')

    # iso2 = PCA(n_components = 2)
    # iso2.fit(Xtrain)
    # Xtrain2 = iso2.fit_transform(Xtrain)
    # Xtest2 = iso2.fit_transform(Xtest)
    # plt.scatter(Xtrain2[:,0], Xtrain2[:,1], c = ytrain)
    # plt.scatter(Xtest2[:,0], Xtest2[:,1], c = yhat, marker = 'X')
    # plt.show()

    #df = pd.read_csv('../poplar_onehot.txt', sep = '\t')
    classes = pd.read_csv('../all_classes.txt', sep = '\t')
    yhat_actual = [LE.classes_[yi] for yi in yhat]
    print(GWsamples)
    print(classes['Geno'])
    metadata = [classes.loc[(classes['Geno'] == gw), 'Full_class'].item() \
        if (gw in classes['Geno'].tolist()) else None for gw in GWsamples]

    #print(metadata)

    gws = pd.DataFrame({'yhat': yhat_actual, 'metadata': metadata}, index = GWsamples)

    gws.to_csv('predicted_GW_samples.csv')

    #print(scores)

if __name__ == '__main__':
    predict()
