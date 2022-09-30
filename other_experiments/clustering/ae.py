import torch

import sys, argparse
sys.path.append('../nn_models/')
from simple_net import *

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

class MashNetAE(torch.nn.Module):
    def __init__(self, input_size, latent_size = 32, hidden_size = 128):
        super(MashNetAE, self).__init__()

        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.encoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.input_size, track_running_stats = False),
            torch.nn.Linear(self.input_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ELU(),
            torch.nn.Linear(128, latent_size)
        )

        # Smaller decoder for computational reasons:
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, input_size),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        return self.decoder(self.encoder(x))

class AE_MashDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        self.X = x.to(device)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx,:]

def train_AE(x, epochs = 100, lr = 0.001, latent_size = 32, batch_size = 32,):

    loader = torch.utils.data.DataLoader(AE_MashDataset(x), \
        batch_size = batch_size, shuffle = True)

    model = MashNetAE(input_size = x.shape[1], latent_size = latent_size)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 0.01)
    criterion = torch.nn.MSELoss()

    cum_loss = []
    for e in range(epochs):
        cum_loss.append(0)
        for Xtrain in loader:
            optimizer.zero_grad()
            x_reconstruct = model(Xtrain)
            loss = criterion(x_reconstruct, Xtrain)
            loss.backward()
            optimizer.step()

            cum_loss[-1] += loss.item()

        if e % 10 == 0:
            print(f'Epoch {e}: Loss = {cum_loss[-1]:.4f}')

    return model

def main(data_path, filter_X = True, fname = 'test'):

    # Gets data and trains autoencoder:
    X, meta = prepare_data(filter_X = filter_X, txt_path = data_path)
    X = torch.as_tensor(X).to(device).float()

    df = pd.read_csv('../all_classes.txt', sep = '\t', index_col = 0)
    #membership = [(df['Geno'].iloc[i] in meta.index.tolist()) for i in range(df.shape[0])]
    meta_classes = df.loc[meta.index,:]

    model = train_AE(X, epochs = 1000, latent_size = 32)
    torch.save(model.state_dict(), '{}.pt'.format(fname))

    # Get encoded:
    Z = model.encoder(X).detach().clone().cpu().numpy()

    T = TSNE().fit_transform(Z)

    cl = LabelEncoder().fit_transform(meta_classes['Full_class'])
    plt.scatter(T[:,0], T[:,1], c = cl)
    plt.savefig(fname + '.png')

    

if __name__ == '__main__':
    # df = pd.read_csv('../all_meta.csv', sep = '\t', index_col = 0)
    # print(df)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mashsize', type = str, default = '4k')
    args = parser.parse_args()

    map_to_mashsize = {
        '500': '500',
        '2k': '2000',
        '4k': '4000',
        '50k': '50000'
    }

    if args.mashsize != '50k':
        data_path = '../data/onehot_s{}.txt'.format(map_to_mashsize[args.mashsize])
        f = True
    else:
        data_path = '../data/trimmed_50000.txt'
        f = False

    main(data_path, filter_X = f, 
        fname = 's{}'.format(map_to_mashsize[args.mashsize]))