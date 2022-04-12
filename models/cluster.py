import sys; sys.path.append('../utils')
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from extract import *

from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph as knn_graph
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

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

    # Create knn graph:
    adj = knn_graph(dist, 4, metric = 'precomputed')

    # Randomly sample nodes:
    random_samp = sorted(random.sample(list(np.arange(adj.shape[0])), 200))

    #adj_samp = adj[random_samp,random_samp]
    
    G = nx.from_numpy_matrix(adj.toarray())

    pos = nx.draw(G, node_color = yordinal, node_size = 200)
    #nx.draw_networkx_labels(G, pos, labels = y)
    plt.show()

    # G = nx.from_numpy_matrix(adj_samp)
    # pos = nx.draw(G, node_color = yordinal[random_samp])
    # plt.show()

    # Overlay with actual labels:

    # Overlay with spectral labels:

if __name__ == '__main__':
    cluster_all()
