from re import L
import torch
import numpy as np
import pandas as pd
import sys; sys.path.append('../utils')
from onehot import *
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score

def prepare_data():
    df = pd.read_csv('../onehot_s4000.txt', sep='\t', index_col=0)
    meta = pd.read_csv('../all_meta.csv', sep='\t', index_col=0)
    
    # Sort based on sample
    # Take union of both
    all_genos = list(set(df.columns.to_list()).intersection(set(meta.index.to_list())))

    X = np.stack([df.loc[:,n].to_numpy() for n in all_genos])

    # Filter X's by low frequency
    exclude = [1, 2, 3, X.shape[0] - 1, X.shape[0]]
    mask = np.ones((X.shape[1],), dtype = bool)
    sum_mat = X.sum(axis=0)
    for i in exclude:
        mask &= (sum_mat != i)

    X = X[:,mask]
    #meta_y = np.stack([meta.loc[n,:] for n in all_genos])
    meta_y = meta.loc[all_genos,:]

    return X, meta_y

def xgboost_predict():
    X, meta = prepare_data()
    y = meta['Longitude']
    model = ElasticNet()
    #model.fit(X, meta['Lattitude'])
    
    scores = cross_val_score(model, X, y, scoring = 'r2', verbose = 2)
    print(scores)

    plt.scatter(meta['Longitude'], meta['Latitude'])
    plt.show()

class MashNet(torch.nn.Module):
    def __init__(self, input_size):
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.PReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.PReLU(),
            torch.nn.Linear(32, 2)
        )
    def forward(x):
        lat, long = self.net(x)
        return {'lat': lat, 'long': long}

class MashDataset(torch.utils.data.Dataset):
    # Dummy dataset to use for dataloaders
    def __init__(self, x, y, device = None):
        self.X = torch.as_tensor(x).to(device)
        self.y = y
        self.device = device
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx,:], \
                self.y['Latitude'].iloc[idx], \
                self.y['Longitude'].iloc[idx]


def train_mashnet(x, y, epochs = 100):
    # Transform x, y to dataloader:
    loader = torch.utils.data.DataLoader(MashDataset(x, y), batch_size = 32)

    model = MashNet(x.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters())

    for e in epochs:
        
        cum_loss = 0

        for x, ylat, ylong in loader:
            out = model(x)
            loss_lat = criterion(out['lat'], ylat)
            loss_long = criterion(out['long'], ylong)
            loss = loss_lat + loss_long
            optimizer.zero_grad()
            loss.backward()
            cum_loss += loss.item()
            optimizer.step()

        if e % 10 == 0:
            print(f'Epoch {e}: Loss = {cum_loss:.4f}')

    return model

def cross_validate():
    pass


if __name__ == '__main__':
    xgboost_predict()
