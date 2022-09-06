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
from sklearn.preprocessing import StandardScaler

from simple_net import prepare_data, prepare_snp_data # Data preppers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MashNet(torch.nn.Module):
    def __init__(self, input_size):
        super(MashNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_size, track_running_stats = False),
            torch.nn.Linear(input_size, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 32),
            torch.nn.LayerNorm(32),
            torch.nn.ELU(),
            torch.nn.Linear(32, 4)
        )
        # self.net = torch.nn.Sequential(
        #     torch.nn.BatchNorm1d(input_size, track_running_stats = False),
        #     torch.nn.Linear(input_size, 128),
        #     torch.nn.ELU(),
        #     torch.nn.Linear(128, 256),
        #     torch.nn.BatchNorm1d(256, track_running_stats = False),
        #     torch.nn.ELU(),
        #     torch.nn.Linear(256, 32),
        #     torch.nn.ELU(),
        #     torch.nn.Linear(32, 4)
        # )
    def forward(self, x):
        pred = self.net(x)
        # Transform based on range
        # lat = pred[:,0].sigmoid() * (60 - 30) + 30 
        # lon = -1.0 * (pred[:,1].sigmoid() * (135 - 90) + 90)
        lat = pred[:,0]
        lon = pred[:,1]
        di = pred[:,2]
        iso = pred[:,3]
        return {'lat': lat, 'long': lon, 'diurinal': di, 'isotherm': iso}

class MashDataset(torch.utils.data.Dataset):
    # Dummy dataset to use for dataloaders
    def __init__(self, x, ylat, ylon, diurinal, isotherm):
        
        self.X = torch.as_tensor(x).to(DEVICE).float()
        self.ylat = torch.as_tensor(ylat).to(DEVICE).float()
        self.ylon = torch.as_tensor(ylon).to(DEVICE).float()
        self.diurinal = torch.as_tensor(diurinal).to(DEVICE).float()
        self.isotherm = torch.as_tensor(isotherm).to(DEVICE).float()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx,:], \
                self.ylat[idx], \
                self.ylon[idx], \
                self.diurinal[idx], \
                self.isotherm[idx] \


def train_mashnet(x, ylat, ylon, diurinal, isotherm, epochs = 300):
    # Transform x, y to dataloader:
    loader = torch.utils.data.DataLoader(MashDataset(x, ylat, ylon, diurinal, isotherm), \
        batch_size = 32, shuffle = True)

    model = MashNet(x.shape[1]).to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.01)

    for e in trange(epochs):
        
        cum_loss = 0

        for xtrain, ylattrain, ylontrain, ditrain, isotrain in loader:
            out = model(xtrain)
            # Calculate all losses
            loss_lat = criterion(out['lat'], ylattrain)
            loss_lon = criterion(out['long'], ylontrain)
            # Need to make this more general for arbitrary climate variable input:
            loss_di = criterion(out['diurinal'], ditrain)
            loss_iso = criterion(out['isotherm'], isotrain)
            loss_add = loss_di + loss_iso

            loss = loss_lat + loss_lon + loss_add # Equal for now
            optimizer.zero_grad()
            loss.backward()
            cum_loss += loss.item()
            optimizer.step()

        if e % 50 == 0:
            print(f'Epoch {e}: Loss = {cum_loss:.4f}')

    return model

def eval_mashnet(model, x, lat, lon, add, ss_lat = None, ss_lon = None, ss_add = None):
    # Convert to tensors:
    x = torch.as_tensor(x).to(DEVICE).float()

    model.eval()
    with torch.no_grad():
        out = model(x)
        if ss_lat is not None:
            out['lat'] = torch.from_numpy(ss_lat.inverse_transform(out['lat'].reshape(-1, 1).detach().cpu()).flatten())
            lat = ss_lat.inverse_transform(lat.reshape(-1, 1)).flatten()
        if ss_lon is not None:
            out['long'] = torch.from_numpy(ss_lon.inverse_transform(out['long'].reshape(-1, 1).detach().cpu()).flatten())
            lon = ss_lon.inverse_transform(lon.reshape(-1, 1)).flatten()

        if ss_add is not None:
            # Construct vect:
            trns = lambda x: x.detach().cpu().flatten()
            #add_mat = np.vstack([trns(out[k]) for k in out.keys() if k not in ['lat', 'long']])
            add_mat = np.vstack([trns(out[k]) for k in ['diurinal', 'isotherm']]).transpose()
            add_mat = torch.from_numpy(ss_add.inverse_transform(add_mat))

    pred = torch.vstack([out['lat'], out['long']])

    lat = torch.as_tensor(lat).cpu().float()
    lon = torch.as_tensor(lon).cpu().float()

    # Compute euclidean distances of error vectors (as tensor operation)
    gt = torch.vstack([lat, lon])

    # print(pred[:5])
    # print('gt', gt[:5])

    # Approximate measure of kilometer error:
    mean_loss = torch.mean(((pred - gt) * 111.139).pow(2).sum(dim=0).sqrt()).item()
    mean_lat_dist = torch.mean((torch.abs(pred - gt)[0,:] * 111.139)).item()
    mean_lon_dist = torch.mean((torch.abs(pred - gt)[1,:] * 111.139)).item()

    # Calculate losses of additional vectors:
    add_losses = []
    for i in range(add_mat.shape[1]):
        loss = (add_mat[:,i] - add[:,i]).pow(2).sum().sqrt().item()
        add_losses.append(loss)
        print(f'Loss {i + 1} = {loss:.4f}')

    return mean_loss, mean_lat_dist, mean_lon_dist, add_losses

def cross_validate(standard_scale_ll = False, additional_keys = ['Diurinal_Range', 'Isothermality']):
    X, meta = prepare_data()
    lat = meta['Latitude'].to_numpy()
    lon = meta['Longitude'].to_numpy()
    add = meta.loc[:,additional_keys].to_numpy()

    kf = KFold(n_splits = 5)
    i = 1
    scores = []
    lat_score = []
    lon_score = []
    add_scores = []
    if standard_scale_ll: 
        ss_lat, ss_lon = StandardScaler(), StandardScaler()
    else:
        ss_lat, ss_lon = None, None 

    ss_add = StandardScaler()

    for train_idx, test_idx in kf.split(X):
        Xtrain, lattrain, lontrain = X[train_idx], lat[train_idx], lon[train_idx]
        Xtest, lattest, lontest = X[test_idx], lat[test_idx], lon[test_idx]

        addtrain, addtest = add[train_idx,:], add[test_idx,:]

        if standard_scale_ll: # Standard scale latitude and longitude
            lattrain, lattest = lattrain.reshape(-1, 1), lattest.reshape(-1, 1)
            ss_lat.fit(lattrain)
            lattrain, lattest = ss_lat.transform(lattrain).flatten(), ss_lat.transform(lattest).flatten()
            lontrain, lontest = lontrain.reshape(-1, 1), lontest.reshape(-1, 1)
            ss_lon.fit(lontrain)
            lontrain, lontest = ss_lon.transform(lontrain).flatten(), ss_lon.transform(lontest).flatten()
    
        # Always standard scale additional features:
        ss_add.fit(addtrain)
        addtrain = ss_add.transform(addtrain)

        # MAKE GENERAL:
        model = train_mashnet(Xtrain, lattrain, lontrain, diurinal = addtrain[:,0], isotherm = addtrain[:,1])

        eval_score, lat_i, lon_i, add_losses = eval_mashnet(model, Xtest, lattest, lontest, 
            add = addtest, ss_lat = ss_lat, ss_lon = ss_lon, ss_add = ss_add)
        print('Split {}: Score = {:.4f}, Lat = {:.4f}, Lon = {:.4f}'.format(i, eval_score, lat_i, lon_i))
        scores.append(eval_score)
        lat_score.append(lat_i)
        lon_score.append(lon_i)
        add_scores.append(add_losses)
        i += 1

    print('Final avg. score: {}'.format(np.mean(scores)))
    print('Final lat score: {}'.format(np.mean(lat_score)))
    print('Final lon score: {}'.format(np.mean(lon_score)))

    add_scores = np.asarray(add_scores)
    for i in range(len(add_scores[0])):
        print(f'Final loss add_{i + 1}: {np.mean(add_scores[:,i])}')

if __name__ == '__main__':
    cross_validate(standard_scale_ll = True)
    #xgboost_predict()

