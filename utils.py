import pickle

import numpy as np
from scipy.sparse import csr_matrix, hstack

def add_bias(X):
    m = X.shape[0]
    if isinstance(X, csr_matrix):
        return hstack([csr_matrix(np.ones((m, 1))), X])
    elif isinstance(X, np.matrix):
        return np.hstack((np.ones((m, 1)), X))
    else:
        raise TypeError('X must be scipy.sparse.csr_matrix or numpy.matrix')

def write_pickled(df, path: str):
    with open(path, mode='wb') as f:
        pickle.dump(df, f)

def load_pickled(path: str):
    with open(path, mode='rb') as f:
        df = pickle.load(f)
    return df
