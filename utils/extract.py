import numpy as np
import pandas as pd
from tqdm import trange, tqdm

def get_pairwise_lookup():
    df = pd.read_csv('../pairwise_dist.txt', sep = '\t', 
        names = ['GENO1', 'GENO2', 'MASH', 'p', 'other'])

    # Create lookup
    lookup = {}

    df_dict = df.to_dict('records')

    for row in tqdm(df_dict):
        key = str(row['GENO1']) + '~' + str(row['GENO2'])
        lookup[key] = row['MASH']

    return lookup

def get_loc_metadata():
    return pd.read_csv('../all_classes.txt', sep = '\t')

def get_climate():
    return pd.read_csv('../climate.txt', sep = '\t')

def get_full_meta():
    # Will have genotypes in index
    return pd.read_csv('../all_meta.csv', sep = '\t', index_col = 0)

def make_distance_matrix(ylist):

    lookup = get_pairwise_lookup()
    num_y = len(ylist)

    lookup_inds = {k:i for i, k in enumerate(ylist)}
    pairwise = np.zeros((num_y, num_y), dtype = float)

    #ymask = {k:True for k in ylist}

    for k, v in tqdm(lookup.items()):
        g1, g2 = k.split('~')
        if (g1 not in lookup_inds):
            #ymask[g1] = False
            continue
        if (g2 not in lookup_inds):
            #ymask[g2] = False
            continue
        if g1 == g2:
            continue
        pairwise[lookup_inds[g1], lookup_inds[g2]] = v

    # Reorder y:
    ymask = (np.sum(pairwise, axis=1) != 0)
    pairwise = pairwise[np.nonzero(ymask)[0],:]
    pairwise = pairwise[:,np.nonzero(ymask)[0]]

    return pairwise, ymask

if __name__ == '__main__':
    # get_pairwise_lookup()
    # get_loc_metadata()
    meta = get_loc_metadata()
    y = meta['Full_class']
    ykey = meta['Geno'].tolist()
    dist, ymask = make_distance_matrix(ykey)

    print(dist.shape)