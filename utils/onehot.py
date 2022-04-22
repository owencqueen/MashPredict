import pandas as pd

from extract import *

# Change this path if you want to run it:
owen_path = '/Users/owenqueen/Desktop/bioinformatics/poplar/MashPredict/poplar_onehot.txt'
def get_onehot(meta, yname = 'Full_class', path = owen_path):
    df = pd.read_csv(path, sep = '\t')
    mat = df.to_numpy()

    # Get keys to positions:
    cnames = df.columns.tolist()
    mapper = {cnames[i]:i for i in range(len(cnames))}

    y = meta[yname]
    gkey = meta['Geno']

    dist, ymask = make_distance_matrix(gkey.tolist())
    # Mask everything:
    gkey = gkey.loc[ymask].tolist()
    y = y.iloc[list(np.nonzero(ymask)[0])].tolist()

    print('len gkey', len(gkey))
    print('len y', len(y))

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