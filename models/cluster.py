from enum import unique
import sys; sys.path.append('../utils')
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams, cycler
from extract import *

from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph as knn_graph
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

def stratify_sampler(dist, y, n = 50):

    y = np.array(y)

    yunique = np.unique(y)

    unique_loc = []

    for yun in yunique:
        yun_args = np.argwhere(y == yun)
        unique_loc.append(list(yun_args))

    minsize = min(min([len(ul) for ul in unique_loc]), n)

    # Sample minsize from all sublists:
    samps = []
    for ul in unique_loc: # Samples from each, concats to list
        samps += list(random.sample(ul, k = minsize))

    return np.array(sorted(samps)).flatten()

def cluster_all():

    meta = get_loc_metadata()
    y = meta['Full_class']
    ykey = meta['Geno'].tolist()
    dist, ymask = make_distance_matrix(ykey)

    y = y.iloc[list(np.nonzero(ymask)[0])].tolist()

    yordinal = LabelEncoder().fit_transform(y)
    print(yordinal[:5])

    model = SpectralClustering(
        n_clusters = 5,
        assign_labels = 'discretize',
        affinity = 'precomputed',
    )

    clustering = model.fit(dist)

    yhat = clustering.labels_

    random_samp = stratify_sampler(dist, y, n = 25)
    lenc = LabelEncoder()
    yordinal = lenc.fit_transform(np.array(y)[random_samp])

    print(lenc.classes_)

    dist_samp = dist[random_samp,:]
    dist_samp = dist_samp[:, random_samp]

    # Create knn graph:
    adj = knn_graph(dist_samp, 2, metric = 'precomputed')

    # Randomly sample nodes:
    #random_samp = sorted(random.sample(list(np.arange(adj.shape[0])), 200))

    #G = nx.from_numpy_matrix(adj.toarray())
    G = nx.from_numpy_matrix(adj.toarray())

    cmap = plt.cm.viridis
    rcParams['axes.prop_cycle'] = cycler(color=cmap(list(range(5))))
    pos = nx.spring_layout(G)
    nx.draw(G, pos = pos, node_color = yordinal, node_size = 200)
    #nx.draw_networkx_labels(G, pos, labels = y)

    cust = [
        Line2D([0], [0], marker = 'o', color = cmap(i/4), label = lenc.classes_[i]) \
            for i in range(5)
    ]

    plt.legend(handles = cust)
    plt.show()

    # G = nx.from_numpy_matrix(adj_samp)
    # pos = nx.draw(G, node_color = yordinal[random_samp])
    # plt.show()

    # Overlay with actual labels:

    # Overlay with spectral labels:

if __name__ == '__main__':
    cluster_all()
