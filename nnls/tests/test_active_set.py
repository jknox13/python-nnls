import pytest
import scipy.optimize as sopt
import scipy.sparse as sp
import numpy as np
from numpy.testing import assert_array_almost_equal

from nnls import block_pivoting, lawson_hanson


def test_block_pivoting():
    # design matrix size (square)
    n = 100

    # ------------------------------------------------------------------------
    # test same output as scipy.nnls
    # A is the n  x n Hilbert matrix
    A = 1. / (np.arange(1, n + 1) + np.arange(n)[:, np.newaxis])
    b = np.ones(n)

    scipy_sol = sopt.nnls(A, b)[0]
    bp_sol = block_pivoting(A, b)

    assert_array_almost_equal(scipy_sol, bp_sol)


def test_lawson_hanson():
    # design matrix size (square)
    n = 100

    # ------------------------------------------------------------------------
    # test same output as scipy.nnls
    # A is the n  x n Hilbert matrix
    A = 1. / (np.arange(1, n + 1) + np.arange(n)[:, np.newaxis])
    b = np.ones(n)

    scipy_sol = sopt.nnls(A, b)[0]
    lh_sol = lawson_hanson(A, b)

    assert_array_almost_equal(scipy_sol, lh_sol)

    # ------------------------------------------------------------------------
    # test sparse
    rng = np.random.RandomState(10293)
    A = rng.randn(n, n)
    idx = rng.choice(np.arange(n**2), int(0.9*n**2), replace=False)
    A[np.unravel_index(idx, (n,n))] = 0

    csr_A = sp.csr_matrix(A.copy())
    csc_A = sp.csc_matrix(A.copy())

    dense_sol = lawson_hanson(A, b)
    csr_sol = lawson_hanson(csr_A, b)
    csc_sol = lawson_hanson(csc_A, b)

    # check cs*_A still sparse
    assert sp.issparse(csr_A)
    assert sp.issparse(csc_A)

    assert_array_almost_equal(csr_sol, dense_sol, 1e-5)
    assert_array_almost_equal(csc_sol, dense_sol, 1e-5)
