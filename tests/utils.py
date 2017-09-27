import os

import pandas as pd
from scipy.sparse import csr_matrix

from winning_price_pred.utils import add_bias


def load_data(fname: str):
    df = pd.read_csv('{}/data/{}'.format(
        os.path.dirname(__file__), fname), header=None)
    X = csr_matrix(df.iloc[:, 0:-1].values)
    y = df.iloc[:, -1].values
    return add_bias(X), y
