import pytest
import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal

from nnls.utils import matrix_rank, solve_lsqr


def test_matrix_rank():
    # ------------------------------------------------------------------------
    # test sparse and dense return same
    rng = np.random.RandomState(10293)
    A = rng.randn(100, 100)
    csr_A = sp.csr_matrix(A.copy())
    csc_A = sp.csc_matrix(A.copy())

    dense_rank = matrix_rank(A)
    csr_rank = matrix_rank(csr_A)
    csc_rank = matrix_rank(csc_A)

    # check cs*_A still sparse
    assert sp.issparse(csr_A)
    assert sp.issparse(csc_A)

    assert_array_almost_equal(csr_A.todense(), csc_A.todense())
    assert_array_almost_equal(A, csr_A.todense())


def test_solve_lsqr():
    # ------------------------------------------------------------------------
    # test sparse and dense return same
    n = 100
    rng = np.random.RandomState(10293)
    A = np.eye(n)
    idx = rng.choice(np.arange(n**2), int(0.9*n**2), replace=False)
    noise = 0.1*rng.rand(n,n)
    noise[np.unravel_index(idx, (n,n))] = 0

    A += noise
    b = rng.randn(n)
    csr_A = sp.csr_matrix(A.copy())
    csc_A = sp.csc_matrix(A.copy())

    dense_sol = solve_lsqr(A, b)
    csr_sol = solve_lsqr(csr_A, b)
    csc_sol = solve_lsqr(csc_A, b)

    # check cs*_A still sparse
    assert sp.issparse(csr_A)
    assert sp.issparse(csc_A)

    assert_array_almost_equal(csr_A.todense(), csc_A.todense())
    assert_array_almost_equal(A, csr_A.todense())
