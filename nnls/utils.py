"""
Utility functions
"""

import numpy as np
from scipy import linalg
from scipy import sparse as sp
from scipy.sparse import linalg as sp_linalg

def solve_lsqr(A, b):
    """Solves least squares problem ||Ax - b||_2^2

    Solves in sparse safe way.

    inspired by sklearn.linear_model.base.LinearRegression.fit()

    Parameters
    ----------
    A
        design matrix
    b
        data vector
    Returns
    -------
    x - solution vector.
    """
    if A.ndim != 2:
        raise ValueError("A must be 2 dimensional")

    if b.ndim != 1:
        raise ValueError("b must be 1 dimensional")

    if sp.issparse(A):
        return sp_linalg.lsqr(A, b)[0]

    return linalg.lsqr(A, b)[0]
