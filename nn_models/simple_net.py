import os
import torch
import numpy as np
import pandas as pd
import sys; sys.path.append('../utils')
from onehot import *
import matplotlib.pyplot as plt
from tqdm import trange

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# lr = 0.001, weight_decay = 0.01
default_opt_kwargs = {
    'lr': 0.01,
    'weight_decay': 0.01
}

def prep(df, meta, geno_axis = 1, filter_X = False):
    
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

def prepare_data(filter_X = True, txt_path = '../data/trimmed_50000.txt'):
    #df = pd.read_csv('../onehot_s50000.txt', sep='\t', index_col=0)
    df = pd.read_csv(txt_path, sep='\t', index_col=0)
    meta = pd.read_csv('../all_meta.csv', sep='\t', index_col=0)
    return prep(df, meta, filter_X = filter_X)

def prepare_snp_data():
    df = pd.read_csv('../aligned_pca_noinland.txt', sep='\t', index_col=0)
    meta = pd.read_csv('../all_meta.csv', sep='\t', index_col=0)
    return prep(df, meta, geno_axis = 0)

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
        # self.net = torch.nn.Sequential(
        #     torch.nn.BatchNorm1d(input_size, track_running_stats = False),
        #     torch.nn.Linear(input_size, 128),
        #     torch.nn.ELU(),
        #     torch.nn.Linear(128, 32),
        #     torch.nn.ELU(),
        #     torch.nn.Linear(32, 2),
        #     torch.nn.ELU(),
        #     torch.nn.Linear(2, 2)
        # )
        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_size, track_running_stats = False),
            torch.nn.Linear(input_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 2)
        )
    def forward(self, x):
        pred = self.net(x)
        # Transform based on range
        # lat = pred[:,0].sigmoid() * (60 - 30) + 30 
        # lon = -1.0 * (pred[:,1].sigmoid() * (135 - 90) + 90)
        lat = pred[:,0]
        lon = pred[:,1]
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


def train_mashnet(x, ylat, ylon, epochs = 500):
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
            loss = loss_lat + loss_lon # Equal weights for now
            optimizer.zero_grad()
            loss.backward()
            cum_loss += loss.item()
            optimizer.step()

        if e % 50 == 0:
            print(f'Epoch {e}: Loss = {cum_loss:.4f}')

    return model

def train_mashnet_w_val(x, ylat, ylon, ss_lat = None, ss_lon = None, epochs = 500, val_pct = 0.1, 
        batch_size = 32, random_state = None, optimizer_kwargs = default_opt_kwargs, mpath = 'tmp.pt',
        rm_mpath = True):
                                       
    # Split out small percentage for validation:
    x_train, x_val, ylat_train, ylat_val, ylon_train, ylon_val = train_test_split(x, ylat, ylon, test_size = val_pct, 
        shuffle = True, random_state = random_state)

    # Transform x, y to dataloader:
    loader = torch.utils.data.DataLoader(MashDataset(x_train, ylat_train, ylon_train), \
        batch_size = batch_size, shuffle = True)

    model = MashNet(x.shape[1]).to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor = 0.1, patience = 50, 
            verbose = True
        )

    best_epoch, best_val_score = 0, 1e9

    for e in trange(epochs):
        
        model.train()
        cum_loss = 0

        for xtrain, ylattrain, ylontrain in loader:
            out = model(xtrain)
            loss_lat = criterion(out['lat'], ylattrain)
            loss_lon = criterion(out['long'], ylontrain)
            loss = loss_lat + loss_lon # Equal weights for now
            optimizer.zero_grad()
            loss.backward()
            cum_loss += loss.item()
            optimizer.step()

        # Eval on validation set:
        val_loss, _, _ = eval_mashnet(model, x_val, ylat_val, ylon_val, ss_lat = ss_lat, ss_lon = ss_lon,
                get_predictions = False)

        scheduler.step(val_loss)

        if val_loss < best_val_score:
            # Save best results:
            best_epoch = e
            # Saves best model to path:
            torch.save(model.state_dict(), mpath)
            best_val_score = val_loss

        if e % 50 == 0:
            print(f'Epoch {e}: Loss = {cum_loss:.4f} \t Val score = {val_loss:.4f}')

    # Loads back in the model with the best performance on the validation set:
    model.load_state_dict(torch.load(mpath))

    if mpath == 'tmp.pt' or rm_mpath:
        os.remove(mpath)

    return model

def eval_mashnet(model, x, lat, lon, ss_lat = None, ss_lon = None, get_predictions = False):
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
    pred = torch.vstack([out['lat'], out['long']])

    lat = torch.as_tensor(lat).cpu().float()
    lon = torch.as_tensor(lon).cpu().float()

    # Compute euclidean distances of error vectors (as tensor operation)
    gt = torch.vstack([lat, lon])

    if get_predictions:
        return gt, pred

    # Approximate measure of kilometer error:
    mean_loss = torch.mean(((pred - gt) * 111.139).pow(2).sum(dim=0).sqrt()).item()
    mean_lat_dist = torch.mean((torch.abs(pred - gt)[0,:] * 111.139)).item()
    mean_lon_dist = torch.mean((torch.abs(pred - gt)[1,:] * 111.139)).item()

    return mean_loss, mean_lat_dist, mean_lon_dist

def cross_validate(standard_scale_ll = False):
    X, meta = prepare_data()
    lat = meta['Latitude'].to_numpy()
    lon = meta['Longitude'].to_numpy()

    kf = KFold(n_splits = 5)
    i = 1
    scores = []
    lat_score = []
    lon_score = []
    if standard_scale_ll: 
        ss_lat, ss_lon = StandardScaler(), StandardScaler()
    else:
        ss_lat, ss_lon = None, None
    for train_idx, test_idx in kf.split(X):
        Xtrain, lattrain, lontrain = X[train_idx], lat[train_idx], lon[train_idx]
        Xtest, lattest, lontest = X[test_idx], lat[test_idx], lon[test_idx]

        if standard_scale_ll: # Standard scale latitude and longitude
            lattrain, lattest = lattrain.reshape(-1, 1), lattest.reshape(-1, 1)
            ss_lat.fit(lattrain)
            lattrain, lattest = ss_lat.transform(lattrain).flatten(), ss_lat.transform(lattest).flatten()
            lontrain, lontest = lontrain.reshape(-1, 1), lontest.reshape(-1, 1)
            ss_lon.fit(lontrain)
            lontrain, lontest = ss_lon.transform(lontrain).flatten(), ss_lon.transform(lontest).flatten()

        model = train_mashnet(Xtrain, lattrain, lontrain)

        eval_score, lat_i, lon_i = eval_mashnet(model, Xtest, lattest, lontest, ss_lat, ss_lon)
        print('Split {}: Score = {:.4f}, Lat = {:.4f}, Lon = {:.4f}'.format(i, eval_score, lat_i, lon_i))
        scores.append(eval_score)
        lat_score.append(lat_i)
        lon_score.append(lon_i)
        i += 1

    print('Final avg. score: {}'.format(np.mean(scores)))
    print('Final lat score: {}'.format(np.mean(lat_score)))
    print('Final lon score: {}'.format(np.mean(lon_score)))

def cross_validate_screen(
        standard_scale_ll = False, 
        n_splits = 5, 
        random_state = None,
        epochs = 800,
        val_pct = 0.1, 
        batch_size = 32, 
        optimizer_kwargs = default_opt_kwargs, 
        mpath_base = None,
        data_path = '../data/trimmed_50000.txt',
        rm_mpath = True,
        ):
    X, meta = prepare_data(txt_path = data_path)
    lat = meta['Latitude'].to_numpy()
    lon = meta['Longitude'].to_numpy()

    kf = KFold(n_splits = 5, shuffle = True, random_state = random_state)
    i = 1
    scores = []
    lat_score = []
    lon_score = []

    pred_gt_pairs = []

    if standard_scale_ll: 
        ss_lat, ss_lon = StandardScaler(), StandardScaler()
    else:
        ss_lat, ss_lon = None, None
    for train_idx, test_idx in kf.split(X):
        Xtrain, lattrain, lontrain = X[train_idx], lat[train_idx], lon[train_idx]
        Xtest, lattest, lontest = X[test_idx], lat[test_idx], lon[test_idx]

        if standard_scale_ll: # Standard scale latitude and longitude
            lattrain, lattest = lattrain.reshape(-1, 1), lattest.reshape(-1, 1)
            ss_lat.fit(lattrain)
            lattrain, lattest = ss_lat.transform(lattrain).flatten(), ss_lat.transform(lattest).flatten()
            lontrain, lontest = lontrain.reshape(-1, 1), lontest.reshape(-1, 1)
            ss_lon.fit(lontrain)
            lontrain, lontest = ss_lon.transform(lontrain).flatten(), ss_lon.transform(lontest).flatten()

        # Set mpath based on split:
        if mpath_base is None:
            mpath = 'tmp.pt'
        else:
            mpath = mpath_base + '_split={}.pt'.format(i)

        model = train_mashnet_w_val(Xtrain, lattrain, lontrain, ss_lat, ss_lon,
            random_state = random_state, epochs = epochs, optimizer_kwargs = optimizer_kwargs,
            batch_size = batch_size, mpath = mpath, rm_mpath = rm_mpath)

        # Get predictions:
        gt, pred = eval_mashnet(model, Xtest, lattest, lontest, ss_lat, ss_lon, get_predictions = True)
        pred_gt_pairs.append((gt, pred))

        eval_score, lat_i, lon_i = eval_mashnet(model, Xtest, lattest, lontest, ss_lat, ss_lon)
        print('Split {}: Score = {:.4f}, Lat = {:.4f}, Lon = {:.4f}'.format(i, eval_score, lat_i, lon_i))
        scores.append(eval_score)
        lat_score.append(lat_i)
        lon_score.append(lon_i)
        i += 1

    print('Final avg. score: {}'.format(np.mean(scores)))
    print('Final lat score: {}'.format(np.mean(lat_score)))
    print('Final lon score: {}'.format(np.mean(lon_score)))

    return pred_gt_pairs, scores, lat_score, lon_score

if __name__ == '__main__':
    #cross_validate(standard_scale_ll = True)
    pred_gt, _, _, _ = cross_validate_screen(standard_scale_ll = True)
    print('pred gt len', len(pred_gt))
    #xgboost_predict()

