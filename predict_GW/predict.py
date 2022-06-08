from tkinter import Label
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

def process():
    df = pd.read_csv('../../data/poplar_onehot.txt', sep = '\t')
    classes = pd.read_csv('../all_meta.csv', sep = '\t')
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

    ytrain = classes['Latitude'].to_numpy()
    ytrain = ytrain[reorder]

    # Split out
    return np.transpose(Xtrain), np.transpose(Xtest), ytrain, df.columns[mask]

# Method to predict from aligned PCA data
def process_aligned():
        df = pd.read_csv('../../data/aligned_pca_noinland.txt', sep = '\t')
        classes = pd.read_csv('../all_meta.csv', sep = '\t')
        lat_labels= classes[['Geno' , 'Latitude']]

        # Convert keys to strings for merging
        df= df.astype({'Geno':'string'})
        lat_labels= lat_labels.astype({'Geno': 'string'})

        # Join labels and data to match order
        labels_pca= pd.merge(lat_labels, df, on='Geno')

        # Select GW samples
        test= labels_pca[labels_pca['Geno'].str.match('GW')]

        # Select all non-GW samples
        df_dup= pd.concat([labels_pca, test])
        train= df_dup.drop_duplicates(keep=False)

        # Return train/test split by labels to main
        return train.drop(columns=['Geno', 'Latitude']), test.drop(columns=['Geno', 'Latitude']), train['Latitude'] , test['Latitude'], test['Geno']

def predict():

    Xtrain, Xtest, ytrain, ytest, GWsamples = process_aligned()

    #OH LE = LabelEncoder()
    #OH ytrain = LE.fit_transform(ytrain)

    #ONEHOT pca = PCA(n_components = 50)
    #ONEHOT pca.fit(Xtrain)

    #ONEHOT Xtrain_pca = pca.transform(Xtrain)
    #ONEHOT Xtest_pca = pca.transform(Xtest)
    Xtrain_pca=Xtrain
    Xtest_pca= Xtest

    model = GradientBoostingRegressor()

    model.fit(Xtrain_pca, ytrain)

    yPred = model.predict(Xtest_pca)
    #print(yhat)
    #scores = cross_val_score(model, X, y, scoring = 'accuracy')

    # iso2 = PCA(n_components = 2)
    # iso2.fit(Xtrain)
    # Xtrain2 = iso2.fit_transform(Xtrain)
    # Xtest2 = iso2.fit_transform(Xtest)
    # plt.scatter(Xtrain2[:,0], Xtrain2[:,1], c = ytrain)
    # plt.scatter(Xtest2[:,0], Xtest2[:,1], c = yhat, marker = 'X')
    # plt.show()

    #df = pd.read_csv('../poplar_onehot.txt', sep = '\t')
    #OH classes = pd.read_csv('../all_meta.csv', sep = '\t')
    #OH yhat_actual = [LE.classes_[yi] for yi in yhat]
    #OH metadata = [classes.loc[(classes['Geno'] == gw), 'Latitude'].item() \
        #if (gw in classes['Geno'].tolist()) else None for gw in GWsamples]

    #print(metadata)

    gws = pd.DataFrame({'Prediction': yPred, 'Metadata': ytest.to_numpy()}, index = GWsamples)
    gws.to_csv('predicted_GW_aligned.csv')

    #print(scores)

if __name__ == '__main__':
    predict()
