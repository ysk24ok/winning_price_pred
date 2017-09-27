import numpy as np
from scipy.sparse import csr_matrix
from nose.tools import assert_equal, assert_true

from winning_price_pred import utils as testee


def test_add_bias():
    # scipy.sparse.csr_matrix
    X = csr_matrix([[1,2],[2,3],[3,4]])
    got = testee.add_bias(X)
    expected = csr_matrix([[1,1,2],[1,2,3],[1,3,4]])
    assert_equal((got - expected).nnz, 0)
    # numpy.matrix
    X = np.matrix([[1,2],[2,3],[3,4]])
    got = testee.add_bias(X)
    expected = np.matrix([[1,1,2],[1,2,3],[1,3,4]])
    assert_true(np.array_equal(got, expected))
