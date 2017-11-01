import os
from typing import Tuple

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

from . import utils


feature_names = (
    'IP', 'Region', 'City', 'AdExchange', 'Domain', 'URL', 'AdSlotId', 'AdSlotWidth',
    'AdSlotHeight', 'AdSlotVisibility', 'AdSlotFormat', 'CreativeID',
    'weekday', 'hour', 'adid', 'usertag', 'ctr', 'wr'
)
#formula = """wr * (IP + Region + City + AdExchange + Domain + URL + AdSlotId +
#AdSlotWidth + AdSlotHeight + AdSlotVisibility + AdSlotFormat + CreativeID +
#weekday + hour + adid + usertag + ctr)"""

mse = make_scorer(mean_squared_error, greater_is_better=False)

vectorized_f = np.vectorize(lambda w, l, is_win: w if is_win is True else l)

def read_csv(path: str):
    """
    First time, read data from csv as dataframe
    From the next time, read data from pickled dataframe
    """
    feature_type_casting = {
        'AdExchange': str,
        'AdSlotId': str,
        'AdSlotWidth': str,
        'AdSlotHeight': str,
        'AdSlotVisibility': str,
        'AdSlotFormat': str,
        'hour': str,
        'adid': str,
        'weekday': str,
        'Region': str,
        'City': str
    }
    path_pickled = '{}.pickle'.format(path)
    # read pickled dataframe from a file if exists
    if os.path.exists(path_pickled):
        return utils.load_pickled(path_pickled)
    df = pd.read_csv(path, index_col=0)
    for colname, type_to_cast in feature_type_casting.items():
        if colname in df.columns:
            df[colname] = df[colname].astype(type_to_cast, copy=False)
    # fill NaN in `IP` and `usertag` column with 'null' if exists
    if 'IP' in df.columns:
        df['IP'].fillna('null', inplace=True)
    if 'usertag' in df.columns:
        df['usertag'].fillna('null', inplace=True)
    # write pickled dataframe to a file
    utils.write_pickled(df, path_pickled)
    return df


def generate_X(
        df: pd.core.frame.DataFrame, feature_names: Tuple[str],
        n_features: int=2**20, add_bias: bool=True) -> csr_matrix:
    D = df.filter(items=feature_names).to_dict(orient='records')
    for d in D:
        # split `usertag` string
        # e.x.
        # {'usertag': '10059,10052,10063'}
        # will become as follows
        # {'usertag=10059': 1, 'usertag=10052': 1, 'usertag=10063': 1}
        if 'usertag' in d:
            for usertag in d['usertag'].split(','):
                d['usertag={}'.format(usertag)] = 1
            # delete original `usertag`
            del d['usertag']
    if add_bias is True:
        X = FeatureHasher(n_features=n_features-1).transform(D)
        X = utils.add_bias(X)
    else:
        X = FeatureHasher(n_features=n_features).transform(D)
    del D
    return X


class BaseLinearModel(BaseEstimator):

    def __init__(
            self, l2reg: float=0.0, tol: float=1e-6, options: dict={}):
        self.l2reg = l2reg
        self.tol = tol
        self.options = options

    def predict(self, X: csr_matrix):
        return X.dot(self.beta)


class LinearModel(BaseLinearModel, RegressorMixin):

    @staticmethod
    def gradient(
            beta: np.ndarray, X: csr_matrix, y: np.ndarray,
            l2reg: float) -> np.ndarray:
        m = X.shape[0]
        z = X.dot(beta) - y
        grad = X.T.dot(z)
        # L2 regularization term
        grad += l2reg * np.append(0, beta[1:])
        grad /= m
        return grad

    @staticmethod
    def loss_function(
            beta: np.ndarray, X: csr_matrix, y: np.ndarray,
            l2reg: float) -> float:
        m = X.shape[0]
        # squared loss
        z = X.dot(beta) - y
        #loss = sum(-norm.logpdf(z))
        loss = sum(z ** 2)
        # L2 regularization term
        loss += l2reg * sum(beta[1:] ** 2)
        loss /= (2 * m)
        return loss

    def fit(
            self, X: csr_matrix, y: np.ndarray,
            initialize_beta_as_zero: bool=False):
        n_features = X.shape[1]
        # initialize beta
        self.beta = np.random.rand(n_features)
        if initialize_beta_as_zero is True:
            self.beta = np.zeros(n_features)
        # optimize
        res = minimize(self.loss_function, self.beta,
            args=(X, y, self.l2reg),
            method='L-BFGS-B',
            jac=self.gradient,
            tol=self.tol,
            options=self.options
        )
        if not res.success:
            raise ValueError("Fitting failed. status: {}, message: {}".format(
                res.status, res.message))
        self.beta = res.x


class CensoredLinearModel(BaseLinearModel, RegressorMixin):

    def __init__(self, is_win: np.ndarray, **kargs):
        self.is_win = is_win
        super(CensoredLinearModel, self).__init__(**kargs)

    @staticmethod
    def gradient(
            beta: np.ndarray, X: csr_matrix, y: np.ndarray,
            is_win: np.ndarray, f, sigma: float, l2reg: float) -> float:
        z = (X.dot(beta) - y) / sigma
        z_lose = -(np.exp(norm.logpdf(z) - norm.logcdf(z)))
        #z_lose = -(norm.pdf(z) / norm.cdf(z))
        z = f(z, z_lose, is_win)
        grad = X.T.dot(z) / sigma
        # L2 regularization term
        grad += l2reg * np.append(0, beta[1:])
        return grad

    @staticmethod
    def loss_function(
            beta: np.ndarray, X: csr_matrix, y: np.ndarray,
            is_win: np.ndarray, f, sigma: float, l2reg: float) -> float:
        z = (X.dot(beta) - y) / sigma
        # loss for win bids
        z_win = -norm.logpdf(z)
        #z_win = -(np.log(1/np.sqrt(2*np.pi)) - z**2/2)
        # loss for lose bids
        z_lose = -norm.logcdf(z)
        loss = sum(f(z_win, z_lose, is_win))
        # L2 regularization term
        loss += l2reg * sum(beta[1:] ** 2) / 2
        return loss

    def fit(
            self, X: csr_matrix, y: np.ndarray,
            initialize_beta_as_zero: bool=False):
        n_features = X.shape[1]
        # initialize beta
        self.beta = np.random.rand(n_features)
        if initialize_beta_as_zero is True:
            self.beta = np.zeros(n_features)
        # optimize
        sigma = np.std(y[self.is_win])
        res = minimize(self.loss_function, self.beta,
            args=(X, y, self.is_win, vectorized_f, sigma, self.l2reg),
            method='L-BFGS-B',
            jac=self.gradient,
            tol=self.tol,
            options=self.options
        )
        if not res.success:
            raise ValueError("Fitting failed. status: {}, message: {}".format(
                res.status, res.message))
        self.beta = res.x


class MixtureModel(object):

    def __init__(self, beta_lm: np.ndarray, beta_clm: np.ndarray):
        self.beta_lm = beta_lm
        self.beta_clm = beta_clm

    def predict(self, X: csr_matrix, wr: np.ndarray):
        return wr * X.dot(self.beta_lm) + (1 - wr) * X.dot(self.beta_clm)


def simulation(
        tr_data_path: str, te_data_path: str, features_names: Tuple[str],
        l2reg_for_lm: float=0.0, l2reg_for_clm: float=0.0,
        n_features=2**20-1, add_bias: bool=True,
        initialize_beta_as_zero: bool=False):
    print('Reading {} for training ...'.format(tr_data_path))
    tr_all_bids = read_csv(tr_data_path)
    print('Reading {} for test ...'.format(te_data_path))
    te_all_bids = read_csv(te_data_path)
    print('Generating X_all for training ...')
    tr_X_all = generate_X(
        tr_all_bids, feature_names, n_features=n_features, add_bias=add_bias)
    tr_y_all = tr_all_bids['PayingPrice'].values
    tr_is_win = tr_all_bids['is_win']
    print('Generating X_win for training ...')
    tr_win_bids = tr_all_bids.query('is_win == True')
    tr_X_win = generate_X(
        tr_win_bids, feature_names, n_features=n_features, add_bias=add_bias)
    tr_y_win = tr_win_bids['PayingPrice'].values
    print('Generating X_all for test ...')
    te_X_all = generate_X(
        te_all_bids, feature_names, n_features=n_features, add_bias=add_bias)
    te_y_all = te_all_bids['PayingPrice']
    te_wr_all = te_all_bids['wr']
    print('Generating X_win for test ...')
    te_win_bids = te_all_bids.query('is_win == True')
    te_X_win = generate_X(
        te_win_bids, feature_names, n_features=n_features, add_bias=add_bias)
    te_y_win = te_win_bids['PayingPrice'].values
    te_wr_win = te_win_bids['wr']
    print('Generating X_lose for test ...')
    te_lose_bids = te_all_bids.query('is_win == False')
    te_X_lose = generate_X(
        te_lose_bids, feature_names, n_features=n_features, add_bias=add_bias)
    te_y_lose = te_lose_bids['PayingPrice'].values
    te_wr_lose = te_lose_bids['wr']
    del tr_all_bids
    del tr_win_bids
    #del te_all_bids
    del te_win_bids
    del te_lose_bids
    print('Fitting LinearModel (l2reg={}) ...'.format(l2reg_for_lm))
    lm = LinearModel(l2reg=l2reg_for_lm)
    lm.fit(tr_X_win, tr_y_win, initialize_beta_as_zero=initialize_beta_as_zero)
    mse_lm_all = -mse(lm, te_X_all, te_y_all)
    mse_lm_win = -mse(lm, te_X_win, te_y_win)
    mse_lm_lose = -mse(lm, te_X_lose, te_y_lose)
    te_all_bids['PredPriceLM'] = lm.predict(te_X_all)
    print('MSE on all: {}, r2score on all: {}'.format(mse_lm_all, lm.score(te_X_all, te_y_all)))
    print('MSE on win: {}, r2score on win: {}'.format(mse_lm_win, lm.score(te_X_win, te_y_win)))
    print('MSE on lose: {}, r2score on lose: {}'.format(mse_lm_lose, lm.score(te_X_lose, te_y_lose)))
    print('Fitting CensoredLinearModel (l2reg={}) ...'.format(l2reg_for_clm))
    clm = CensoredLinearModel(tr_is_win, l2reg=l2reg_for_clm)
    clm.fit(tr_X_all, tr_y_all, initialize_beta_as_zero=initialize_beta_as_zero)
    mse_clm_all = -mse(clm, te_X_all, te_y_all)
    mse_clm_win = -mse(clm, te_X_win, te_y_win)
    mse_clm_lose = -mse(clm, te_X_lose, te_y_lose)
    te_all_bids['PredPriceCLM'] = clm.predict(te_X_all)
    print('MSE on all: {}, r2score on all: {}'.format(mse_clm_all, clm.score(te_X_all, te_y_all)))
    print('MSE on win: {}, r2score on win: {}'.format(mse_clm_win, clm.score(te_X_win, te_y_win)))
    print('MSE on lose: {}, r2score on lose: {}'.format(mse_clm_lose, clm.score(te_X_lose, te_y_lose)))
    print('Predicting by MixtureModel...')
    mix = MixtureModel(lm.beta, clm.beta)
    te_y_all_pred = mix.predict(te_X_all, te_wr_all)
    te_y_win_pred = mix.predict(te_X_win, te_wr_win)
    te_y_lose_pred = mix.predict(te_X_lose, te_wr_lose)
    mse_mix_all = mean_squared_error(te_y_all, te_y_all_pred)
    mse_mix_win = mean_squared_error(te_y_win, te_y_win_pred)
    mse_mix_lose = mean_squared_error(te_y_lose, te_y_lose_pred)
    te_all_bids['PredPriceMix'] = te_y_all_pred
    print('MSE on all: {}, r2score on all: {}'.format(mse_mix_all, r2_score(te_y_all, te_y_all_pred)))
    print('MSE on win: {}, r2score on win: {}'.format(mse_mix_win, r2_score(te_y_win, te_y_win_pred)))
    print('MSE on lose: {}, r2score on lose: {}'.format(mse_mix_lose, r2_score(te_y_lose, te_y_lose_pred)))
    return te_all_bids, [mse_lm_all, mse_lm_win, mse_lm_lose, mse_clm_all, mse_clm_win, mse_clm_lose, mse_mix_all, mse_mix_win, mse_mix_lose]


if __name__ == '__main__':
    import sys
    from datetime import datetime, timedelta

    if len(sys.argv) != 2:
        sys.exit('Usage: python -m winning_price_pred.censored_reg 20130606')
    tr_yyyymmdd = sys.argv[1]
    te_datetime = datetime.strptime(tr_yyyymmdd, '%Y%m%d') + timedelta(days=1)
    te_yyyymmdd = te_datetime.strftime('%Y%m%d')
    d = os.path.dirname(os.path.abspath(__file__))
    tr_data_path = '{}/data/bidimpclk.{}.sim2.csv'.format(d, tr_yyyymmdd)
    te_data_path = '{}/data/bidimpclk.{}.sim2.csv'.format(d, te_yyyymmdd)

    pd.set_option('display.width', 160)

    df, mses = simulation(
        tr_data_path, te_data_path, feature_names,
        l2reg_for_lm=10, l2reg_for_clm=1)
    print()
    print(df.head(n=20).filter(['BiddingPrice', 'NewBiddingPrice', 'PayingPrice', 'is_win', 'PredPriceLM', 'PredPriceCLM', 'PredPriceMix']))
