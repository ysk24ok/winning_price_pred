import os
from typing import Tuple

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import r2_score

from . import utils


feature_names = (
    'IP', 'Region', 'City', 'AdExchange', 'Domain', 'URL', 'AdSlotId', 'AdSlotWidth',
    'AdSlotHeight', 'AdSlotVisibility', 'AdSlotFormat', 'CreativeID',
    'weekday', 'hour', 'adid', 'usertag', 'ctr', 'wr'
)
#formula = """wr * (IP + Region + City + AdExchange + Domain + URL + AdSlotId +
#AdSlotWidth + AdSlotHeight + AdSlotVisibility + AdSlotFormat + CreativeID +
#weekday + hour + adid + usertag + ctr)"""


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
    df = df.astype(feature_type_casting, copy=False)
    # fill NaN in `IP` and `usertag` column with 'null'
    df['IP'].fillna('null', inplace=True)
    df['usertag'].fillna('null', inplace=True)
    # write pickled dataframe to a file
    utils.write_pickled(df, path_pickled)
    return df


def generate_hashed_X(
        df: pd.core.frame.DataFrame, feature_names: Tuple[str]) -> csr_matrix:
    filtered_df = df.filter(items=feature_names)
    D = filtered_df.to_dict(orient='records')
    del filtered_df
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
    return FeatureHasher().transform(D)


class BaseLinearModel(BaseEstimator):

    def __init__(
            self, fit_intercept: bool=True, l2reg: float=0.0, tol: float=1e-3,
            options: dict={}):
        self.fit_intercept = fit_intercept
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
        # weight for bias term is not included
        grad += l2reg * np.append(0, beta[1:])
        #grad += l2reg * np.append(1, beta[1:])
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
        # weight for bias term is not included
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

    @staticmethod
    def gradient(
            beta: np.ndarray, X_win: csr_matrix, y_win: np.ndarray,
            X_lose: csr_matrix, y_lose: np.ndarray,
            sigma: float, l2reg: float) -> float:
        # gradient for win bids
        z_win = (X_win.dot(beta) - y_win) / sigma
        grad = X_win.T.dot(z_win) / sigma
        # gradient for lose bids
        z_lose = (X_lose.dot(beta) - y_lose) / sigma
        #grad += -X_lose.T.dot(norm.pdf(z_lose) / norm.cdf(z_lose))
        grad += -X_lose.T.dot(np.exp(norm.logpdf(z_lose) - norm.logcdf(z_lose))) / sigma
        # L2 regularization term
        grad += l2reg * np.append(0, beta[1:]) * 2
        #grad += l2reg * np.append(1, beta[1:]) * 2
        return grad

    @staticmethod
    def loss_function(
            beta: np.ndarray, X_win: csr_matrix, y_win: np.ndarray,
            X_lose: csr_matrix, y_lose: np.ndarray,
            sigma: float, l2reg: float) -> float:
        # loss for win bids
        z_win = (X_win.dot(beta) - y_win) / sigma
        loss = sum(-norm.logpdf(z_win))
        #loss = sum(-np.log(1/np.sqrt(2*np.pi)) + z_win ** 2 / 2)
        # loss for lose bids
        z_lose = (X_lose.dot(beta) - y_lose) / sigma
        loss += sum(-norm.logcdf(z_lose))
        # L2 regularization term
        loss += l2reg * sum(beta[1:] ** 2)
        return loss

    def fit(
            self, X_win: csr_matrix, y_win: np.ndarray,
            X_lose: csr_matrix, y_lose: np.ndarray,
            initialize_beta_as_zero: bool=False):
        n_features = X_win.shape[1]
        # initialize beta
        self.beta = np.random.rand(n_features)
        if initialize_beta_as_zero is True:
            self.beta = np.zeros(n_features)
        # optimize
        sigma = np.std(y_win)
        res = minimize(self.loss_function, self.beta,
            args=(X_win, y_win, X_lose, y_lose, sigma, self.l2reg),
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


if __name__ == '__main__':
    import sys
    from datetime import datetime, timedelta
    from sklearn.metrics import mean_squared_error

    if len(sys.argv) != 2:
        sys.exit('Usage: python -m winning_price_pred.censored_reg 20130606')
    tr_yyyymmdd = sys.argv[1]
    te_datetime = datetime.strptime(tr_yyyymmdd, '%Y%m%d') + timedelta(days=1)
    te_yyyymmdd = te_datetime.strftime('%Y%m%d')
    d = os.path.dirname(os.path.abspath(__file__))
    tr_data_path = '{}/data/bidimpclk.{}.sim2.csv'.format(d, tr_yyyymmdd)
    te_data_path = '{}/data/bidimpclk.{}.sim2.csv'.format(d, te_yyyymmdd)

    print('Reading {} for training ...'.format(tr_data_path))
    tr_all_bids = read_csv(tr_data_path)
    print('Reading {} for test ...'.format(te_data_path))
    te_all_bids = read_csv(te_data_path)
    print('Generating win bids for training ...')
    tr_win_bids = tr_all_bids.query("is_win == True")
    tr_X_win = utils.add_bias(generate_hashed_X(tr_win_bids, feature_names))
    tr_y_win = tr_win_bids["PayingPrice"].values
    print('Generating lose bids for training ...')
    tr_lose_bids = tr_all_bids.query("is_win == False")
    tr_X_lose = utils.add_bias(generate_hashed_X(tr_lose_bids, feature_names))
    tr_y_lose = tr_lose_bids["NewBiddingPrice"].values
    print('Generating all bids for test ...')
    te_X = utils.add_bias(generate_hashed_X(te_all_bids, feature_names))
    te_y = te_all_bids['PayingPrice']
    te_wr = te_all_bids['wr']
    print('Fitting LinearModel ...')
    lm = LinearModel()
    lm.fit(tr_X_win, tr_y_win)
    print('Fitting CensoredLinearModel ...')
    clm = CensoredLinearModel()
    clm.fit(tr_X_win, tr_y_win, tr_X_lose, tr_y_lose)
    print('Predicting by LinearModel...')
    print('MSE: {}, r2score: {}'.format(
        mean_squared_error(te_y, lm.predict(te_X)), lm.score(te_X, te_y)))
    print('Predicting by CensoredLinearModel...')
    print('MSE: {}, r2score: {}'.format(
        mean_squared_error(te_y, clm.predict(te_X)), clm.score(te_X, te_y)))
    print('Predicting by MixtureModel...')
    mix = MixtureModel(lm.beta, clm.beta)
    te_y_pred = mix.predict(te_X, te_wr)
    print('MSE: {}, r2score: {}'.format(
        mean_squared_error(te_y, te_y_pred), r2_score(te_y, te_y_pred)))
