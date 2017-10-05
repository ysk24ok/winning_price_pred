from unittest import TestCase

import numpy as np
from scipy.optimize import check_grad
from nose.tools import assert_almost_equal

from winning_price_pred import censored_reg as testee
from winning_price_pred.tests.utils import load_data


class TestLinearModel(TestCase):

    def setUp(self):
        X, y = load_data('ex1data1.txt')
        self.X = X
        self.y = y
        self.n_features = X.shape[1]

    def test_loss_function(self):
        lm = testee.LinearModel(self.n_features)
        got = lm.loss_function(lm.beta, self.X, self.y, lm.l2reg)
        assert_almost_equal(got, 32.0727, places=4)

    def test_loss_function_with_l2reg(self):
        lm = testee.LinearModel(self.n_features, l2reg=1)
        lm.beta = np.ones(self.n_features)
        got = lm.loss_function(lm.beta, self.X, self.y, lm.l2reg)
        assert_almost_equal(got, 10.2717, places=4)

    def test_gradient(self):
        lm = testee.LinearModel(self.n_features)
        lm.initialize_beta()
        got = check_grad(lm.loss_function, lm.gradient, lm.beta, self.X, self.y, lm.l2reg)
        assert_almost_equal(got, 0.0000, places=4)

    def test_gradient_with_l2reg(self):
        lm = testee.LinearModel(self.n_features, l2reg=1)
        lm.initialize_beta()
        #lm.beta = np.ones(self.n_features)
        got = check_grad(lm.loss_function, lm.gradient, lm.beta, self.X, self.y, lm.l2reg)
        assert_almost_equal(got, 0.0000, places=4)


class TestCensoredLinearModel(TestCase):

    def setUp(self):
        X_win, y_win = load_data('ex1data1.txt')
        X_lose, y_lose = load_data('ex1data1.txt')
        self.X_win = X_win
        self.X_lose = X_lose
        self.y_win = y_win
        self.y_lose = y_lose
        self.sigma = np.std(y_win)
        self.n_features = X_win.shape[1]

    def test_loss_function(self):
        clm = testee.CensoredLinearModel(self.n_features)
        got = clm.loss_function(
            clm.beta, self.X_win, self.y_win, self.X_lose, self.y_lose,
            self.sigma, clm.l2reg)
        assert_almost_equal(got, 421.7217, places=4)

    def test_loss_function_with_l2reg(self):
        clm = testee.CensoredLinearModel(self.n_features)
        clm.beta = np.ones(self.n_features)
        got = clm.loss_function(
            clm.beta, self.X_win, self.y_win, self.X_lose, self.y_lose,
            self.sigma, clm.l2reg)
        assert_almost_equal(got, 161.3537, places=4)

    def test_gradient(self):
        clm = testee.CensoredLinearModel(self.n_features)
        clm.initialize_beta()
        got = check_grad(
            clm.loss_function, clm.gradient, clm.beta,
            self.X_win, self.y_win, self.X_lose, self.y_lose,
            self.sigma, clm.l2reg)
        assert_almost_equal(got, 0.000, places=3)

    def test_gradient_with_l2reg(self):
        clm = testee.CensoredLinearModel(self.n_features, l2reg=1)
        clm.initialize_beta()
        got = check_grad(
            clm.loss_function, clm.gradient, clm.beta,
            self.X_win, self.y_win, self.X_lose, self.y_lose,
            self.sigma, clm.l2reg)
        assert_almost_equal(got, 0.000, places=3)
