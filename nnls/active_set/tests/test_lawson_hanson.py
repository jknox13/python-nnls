import pytest
import scipy.optimize as sopt
import numpy as np
from numpy.testing import assert_allclose

from nnls import lawson_hanson


def test_lawson_hanson():
    # ------------------------------------------------------------------------
    # test same output as scipy.nnls
    # A is the n  x n Hilbert matrix
    n = 100
    A = 1. / (np.arange(1, n + 1) + np.arange(n)[:, np.newaxis])
    b = np.ones(n)

    scipy_sol = sopt.nnls(A, b)[0]
    lh_sol = lawson_hanson(A, b)

    assert_allclose(scipy_sol, lh_sol)
