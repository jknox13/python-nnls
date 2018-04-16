"""
Utility functions
"""

import numpy as np
from scipy import linalg
from scipy import sparse as sp
from scipy.sparse import linalg as sp_linalg

def matrix_rank(A, tol=1e-12):
    """Returns the matrix rank of A.

    Returns in sparse safe way.

    Parameters
    ----------
    A
        design matrix
    """
    if A.ndim != 2:
        raise ValueError("A must be 2 dimensional")

    if sp.issparse(A):
        s = sp_linalg.svds(A, k=min(A.shape)-1, return_singular_vectors=False)
    else:
        s = linalg.svd(A, compute_uv=False, overwrite_a=False)

    return np.sum(s > tol)



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

    return linalg.lstsq(A, b)[0]
