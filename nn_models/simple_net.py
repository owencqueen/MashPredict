from re import L
import torch
import numpy as np
import pandas as pd
import sys; sys.path.append('../utils')
from onehot import *
import matplotlib.pyplot as plt
from tqdm import trange

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, KFold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    y = meta['Latitude']
    model = ElasticNet()
    #model.fit(X, meta['Lattitude'])
    
    scores = cross_val_score(model, X, y, scoring = 'neg_mean_squared_error', verbose = 2)
    print(scores)

    plt.scatter(meta['Longitude'], meta['Latitude'])
    plt.show()

class MashNet(torch.nn.Module):
    def __init__(self, input_size):
        super(MashNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.PReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.PReLU(),
            torch.nn.Linear(32, 2)
        )
    def forward(self, x):
        pred = self.net(x)
        # Transform based on range
        lat = pred[:,0].sigmoid() * (60 - 30) + 30 
        lon = -1.0 * (pred[:,1].sigmoid() * (135 - 90) + 90)
        return {'lat': lat, 'long': lon}

class MashDataset(torch.utils.data.Dataset):
    # Dummy dataset to use for dataloaders
    def __init__(self, x, ylat, ylon):
        
        self.X = torch.as_tensor(x).to(DEVICE).float()
        self.ylat = torch.as_tensor(ylat).to(DEVICE).float()
        self.ylon = torch.as_tensor(ylon).to(DEVICE).float()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx,:], \
                self.ylat[idx], \
                self.ylon[idx]


def train_mashnet(x, ylat, ylon, epochs = 300):
    # Transform x, y to dataloader:
    loader = torch.utils.data.DataLoader(MashDataset(x, ylat, ylon), \
        batch_size = 32, shuffle = True)

    model = MashNet(x.shape[1]).to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.01)

    for e in trange(epochs):
        
        cum_loss = 0

        for xtrain, ylattrain, ylontrain in loader:
            out = model(xtrain)
            loss_lat = criterion(out['lat'], ylattrain)
            loss_lon = criterion(out['long'], ylontrain)
            loss = loss_lat + loss_lon # Equal for now
            optimizer.zero_grad()
            loss.backward()
            cum_loss += loss.item()
            optimizer.step()

        if e % 50 == 0:
            print(f'Epoch {e}: Loss = {cum_loss:.4f}')

    return model

def eval_mashnet(model, x, lat, lon):
    # Convert to tensors:
    x = torch.as_tensor(x).to(DEVICE).float()
    lat = torch.as_tensor(lat).to(DEVICE).float()
    lon = torch.as_tensor(lon).to(DEVICE).float()

    model.eval()
    out = model(x)
    pred = torch.vstack([out['lat'], out['long']])

    # Compute euclidean distances of error vectors (as tensor operation)
    gt = torch.vstack([lat, lon])
    # Approximate measure of kilometer error:
    mean_loss = torch.mean(((pred - gt) * 111.139).pow(2).sum(dim=0).sqrt()).item()
    mean_lat_dist = torch.mean((torch.abs(pred - gt)[0,:] * 111.139)).item()
    mean_lon_dist = torch.mean((torch.abs(pred - gt)[1,:] * 111.139)).item()

    return mean_loss, mean_lat_dist, mean_lon_dist

def cross_validate():
    X, meta = prepare_data()
    lat = meta['Latitude'].to_numpy()
    lon = meta['Longitude'].to_numpy()

    kf = KFold(n_splits = 5)
    i = 1
    scores = []
    lat_score = []
    lon_score = []
    for train_idx, test_idx in kf.split(X):
        Xtrain, lattrain, lontrain = X[train_idx], lat[train_idx], lon[train_idx]
        Xtest, lattest, lontest = X[test_idx], lat[test_idx], lon[test_idx]
        model = train_mashnet(Xtrain, lattrain, lontrain)

        eval_score, lat_i, lon_i = eval_mashnet(model, Xtest, lattest, lontest)
        print('Split {}: Score = {:.4f}, Lat = {:.4f}, Lon = {:.4f}'.format(i, eval_score, lat_i, lon_i))
        scores.append(eval_score)
        lat_score.append(lat_i)
        lon_score.append(lon_i)
        i += 1

    print('Final avg. score: {}'.format(np.mean(scores)))
    print('Final lat score: {}'.format(np.mean(lat_score)))
    print('Final lon score: {}'.format(np.mean(lon_score)))

if __name__ == '__main__':
    cross_validate()
    #xgboost_predict()

