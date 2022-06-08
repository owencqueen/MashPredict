import pandas as pd

from extract import *

# Change this path if you want to run it:
owen_path = '../data/poplar_onehot.txt'
def get_onehot(meta, yname = 'Latitude', path = owen_path, regression = True):
    df = pd.read_csv(path, sep = '\t')
    mat = df.to_numpy()

    # Get keys to positions:
    cnames = df.columns.tolist()
    mapper = {cnames[i]:i for i in range(len(cnames))}

    y = meta[yname]
    gkey = meta['Geno'] if ('Geno' in meta.columns) else pd.Series(meta.index)

    dist, ymask = make_distance_matrix(gkey.tolist())
    ymask = np.array(ymask)
    if regression:
        ymask &= ~np.isnan(y.to_numpy()) # Mask out those that are NaN
    # Mask everything:
    gkey = gkey.loc[ymask].tolist()
    y = y.iloc[list(np.nonzero(ymask)[0])].tolist()

    filtermat = np.empty((len(y), mat.shape[0]))

    i = 0
    for g in gkey:
        ind = mapper[g]
        # Fill in place:
        filtermat[i,:] = mat[:,ind]
        i += 1

    # print('FM shape', filtermat.shape)
    # exit()

    return filtermat, y, mapper
