import sys; sys.path.append('/data1/compbio/oqueen/poplar/MashPredict/utils')
import numpy as np
import pandas as pd
import networkx as nx

from extract import *

from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph as knn_graph
from sklearn.model_selection import cross_val_score

def cluster_all():

    meta = get_loc_metadata()
    y = meta['Full_class']
    ykey = meta['Geno'].tolist()
    dist, ymask = make_distance_matrix(ykey)

    y = y.iloc[list(np.nonzero(ymask)[0])].tolist()

    model = SpectralClustering(
        n_clusters = 5,
        assign_labels = 'discretize',
        metric = 'precomputed',
    )

    clustering = model.fit(dist)

    yhat = clustering.labels_

    # Create knn graph:
    adj = knn_graph(dist, metric = 'precomputed')
    
    G = nx.from_numpy_matrix(adj)



    nx.draw(G)
    plt.show()

    # Overlay with actual labels:

    # Overlay with spectral labels:

if __name__ == '__main__':
    cluster_all()