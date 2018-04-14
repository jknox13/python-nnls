"""
Lawson-Hanson algorithm for computing the nonnegative least squares solution.
"""

from __future__ import division

import numpy as np

from ..solvers import solve_lsqr

# TODO : boolean arrays

def lawson_hanson(A, b, tol=1e-3, maxiter=np.inf):
    """Lawson-Hanson algorithm for computing the nonnegative least squares solution.

    The Lawson-Hanson algorithm modifies the active set by one element in each
    step. It combines forward selection with backwards steps that reduce the
    active set in case the estimate is not feasible in the active set.

    In general no upper bound on iterations. Worst case : 2^p (all possible
    subsets)

    MATLAB package :: lsqnonneg

    use Martin Slawski's rank-one-updates of Cholesky/QR to reduce complexity
    from O(|S|^3) to O(|S|^2) (also in ref [47])

    Parameters
    ----------
    A

    b

    Examples
    --------

    References
    ----------
    [37]
    Algorithm : [12]
    """
    n, m = A.shape

    # compute gram matrix
    Q = A.T.dot(A)
    c = A.T.dot(b)

    # initialize
    n_iter = 0
    k = 0
    x = np.zeros(m)
    r = c.copy()
    active_set = list(range(m))
    passive_set = []

    while active_set and n_iter < maxiter:
        #k += 1

        w = A.T.dot(r)[active_set]
        if w > tol:
            break

        # update sets
        j = np.argmax(w)
        active_set.remove(j)
        passive_set.append(j)

        # TODO: incorp cholesky updates
        x_new = np.zeros(m)
        x_new[passive_set] = solve_lsqr(A[passive_set], b)

        while np.any(x_new <= tol):
            # remove elements elements which no longer belong
            q_set = list(set(np.where(x_new <= tol)[0]) and set(passive_set))

            alpha = np.zeros(m)
            alpha[q_set] = x[q_set] / (x[q_set] - x_new[q_set])

            alpha = alpha[alpha.nonzero()].min()

            x = x + alpha*(x_new - x)





            # TODO: incorp cholesky updates





        # update iter count
        n_iter += 1



