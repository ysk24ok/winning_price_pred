import os
from unittest import TestCase

import numpy as np
from scipy.optimize import check_grad
from nose.tools import assert_almost_equal

from winning_price_pred import censored_reg as testee


def load_data_as_df(fname: str):
    data_path = '{}/data/{}'.format(os.path.dirname(__file__), fname)
    return testee.read_csv(data_path)


class TestLinearModel(TestCase):

    def setUp(self):
        feature_names = ('Domain', 'AdSlotWidth', 'AdSlotHeight')
        df = load_data_as_df('sample.csv')
        is_win = df['is_win']
        self.X = testee.generate_X(df[is_win], feature_names, n_features=50)
        self.y = df['BiddingPrice'][is_win].values
        self.n_features = self.X.shape[1]

    def test_gradient(self):
        lm = testee.LinearModel()
        init_beta = np.random.rand(self.n_features)
        got = check_grad(
            lm.loss_function, lm.gradient, init_beta, self.X, self.y, lm.l2reg)
        assert_almost_equal(got, 0, places=2)

    def test_gradient_with_l2reg(self):
        lm = testee.LinearModel(l2reg=1)
        init_beta = np.random.rand(self.n_features)
        got = check_grad(
            lm.loss_function, lm.gradient, init_beta, self.X, self.y, lm.l2reg)
        assert_almost_equal(got, 0, places=2)


class TestCensoredLinearModel(TestCase):

    def setUp(self):
        feature_names = ('Domain', 'AdSlotWidth', 'AdSlotHeight')
        df = load_data_as_df('sample.csv')
        self.is_win = df['is_win']
        self.X = testee.generate_X(df, feature_names, n_features=50)
        self.y = testee.vectorized_f(
            df['NewBiddingPrice'], df['PayingPrice'], self.is_win)
        self.n_features = self.X.shape[1]
        self.sigma = np.std(df['PayingPrice'][self.is_win])

    def test_gradient(self):
        clm = testee.CensoredLinearModel(self.is_win)
        init_beta = np.zeros(self.n_features)
        got = check_grad(
            clm.loss_function, clm.gradient, init_beta,
            self.X, self.y, self.is_win, testee.vectorized_f,
            self.sigma, clm.l2reg)
        assert_almost_equal(got, 0, places=3)

    def test_gradient_with_l2reg(self):
        clm = testee.CensoredLinearModel(self.is_win, l2reg=1)
        init_beta = np.random.rand(self.n_features)
        got = check_grad(
            clm.loss_function, clm.gradient, init_beta,
            self.X, self.y, self.is_win, testee.vectorized_f,
            self.sigma, clm.l2reg)
        assert_almost_equal(got, 0, places=3)
