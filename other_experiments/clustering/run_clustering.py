import torch
import sys, argparse
import numpy as np
sys.path.append('../nn_models/')
sys.path.append('../utils')
from extract import *
from simple_net import *

from ae import MashNetAE
from vae import MashNetVAE

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score as MI
from sklearn.model_selection import GridSearchCV

device = "cuda" if torch.cuda.is_available() else "cpu"

map_to_mashsize = {
    '500': '500',
    '2k': '2000',
    '4k': '4000',
    '50k': '50000'
}

def cluster_AE(X, y, method = 'dbscan'):

    ylab = LabelEncoder().fit_transform(y)

    if method == 'kmeans':
        kvals = np.arange(30, 888)
        score = 0
        for k in kvals:
            model = KMeans(n_clusters = k)
            labels = model.fit_predict(X)

            # Cluster evaluation:
            s = MI(ylab, labels)
            print('s = {:.4f}, k = {}, nc = {}'.format(s, k, len(np.unique(labels))))
            score = score if score > s else s
    else:
        # Search over epsilon values:
        eps_grid = np.concatenate([np.geomspace(1e-20,1e-10,num=10)])
        min_points = np.arange(0, 10)
        score = 0
        for eps in eps_grid:
            for m in min_points:
                model = DBSCAN(eps = eps, min_samples = m)
                labels = model.fit_predict(X)
                s = MI(ylab, labels)
                print('s = {:.4f}, m = {}, eps = {}, nc = {}'.format(s, m, eps, len(np.unique(labels))))
                score = score if score > s else s

    
    print('Score: {:.4f}'.format(score))
    print('Null Score: {:.4f}'.format(MI(np.arange(len(ylab)), ylab)))


def cluster_dist(dist, y, method = 'dbscan'):

    ylab = LabelEncoder().fit_transform(y)

    if method == 'kmeans':
        model = KMeans(n_clusters = len(np.unique(ylab)), metric = 'precomputed')
    else:
        # Search over epsilon values:
        eps_grid = np.concatenate([np.geomspace(1e-20,1e-10,num=9)])
        #eps_grid = np.linspace(1e-1, 1, num=11)
        min_points = np.arange(0, 10)
        score = 0
        for eps in eps_grid:
            for m in min_points:
                model = DBSCAN(eps = eps, min_samples = m)
                labels = model.fit_predict(dist)
                s = MI(ylab, labels)
                print('s = {:.4f}, m = {}, eps = {}, nc = {}'.format(s, m, eps, len(np.unique(labels))))
                score = score if score > s else s

    #labels = model.fit_predict(dist)

    # Cluster evaluation:
    #score = MI(ylab, labels)
    print('Score: {:.4f}'.format(score))

def get_ae_embeddings(args, X, vae = False):

    if vae:
        model = MashNetVAE(input_size = X.shape[-1])
        model.to(device)
        
        fname = 'vae_s{}.pt'.format(map_to_mashsize[args.mashsize])
        model.load_state_dict(torch.load(fname))

    else:
        model = MashNetAE(input_size = X.shape[-1])
        model.to(device)
        
        fname = 's{}.pt'.format(map_to_mashsize[args.mashsize])
        model.load_state_dict(torch.load(fname))

    Z = model.encoder(X)

    z = Z.detach().clone().cpu().numpy()

    return z

def get_dist_matrix():
    pass

def align_meta(align_from, align_to):
    # Finds all samples in align_from that are in align_to, outputs River system for them
    river_y = align_from.index.map(align_to['Full_class'])
    xmask = river_y.notna()
    return xmask, river_y[xmask]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mashsize', type = str, default = '4k')
    parser.add_argument('--embed', type = str)
    parser.add_argument('--method', type = str)
    parser.add_argument('--vae', action = 'store_true')
    args = parser.parse_args()

    assert args.embed in ['dist', 'ae']
    # dist: use distance metric
    # ae: use AE embeddings

    assert args.method in ['dbscan', 'kmeans']

    filter_X = (args.mashsize != '50k')

    #meta_classes = pd.read_csv('../poplar_river_trim.csv', index_col = 0)
    meta_classes = pd.read_csv('../all_classes.txt', sep = '\t', index_col = 0)
    
    # Load in X, y:
    if args.embed == 'ae':
        if args.mashsize != '50k':
            data_path = '../data/onehot_s{}.txt'.format(map_to_mashsize[args.mashsize])
            f = True
        else:
            data_path = '../data/trimmed_50000.txt'
            f = False
        X, meta = prepare_data(filter_X = f, txt_path = data_path)

        xmask, y = align_meta(meta, meta_classes)
        y = y.tolist()

        X = X[xmask,:]
        X = torch.as_tensor(X).to(device).float()

        z = get_ae_embeddings(args, X, vae = args.vae)
        cluster_AE(z, y, method = args.method)

    elif args.embed == 'dist':
        meta = pd.read_csv('/lustre/isaac/scratch/oqueen/MashPredict/all_meta.csv', sep = '\t', index_col = 0)
        dist_path = '../data/s{}_dist.txt'.format(map_to_mashsize[args.mashsize])

        ykey = meta.index.tolist()
        dist, ymask = make_distance_matrix(ykey, data_path = dist_path)
        meta = meta.iloc[np.nonzero(ymask)[0],:]

        # Align meta:
        xmask, y = align_meta(meta, meta_classes)
        y = y.tolist()
        D = dist[xmask,:][:,xmask]
        cluster_dist(D, y, method = args.method)





