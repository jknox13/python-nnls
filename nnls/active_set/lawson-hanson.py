"""
Lawson-Hanson algorithm for computing the nonnegative least squares solution.
"""

# Authors: Joseph Knox <joseph.edward.knox@gmail.com>
# License: BSD 3

from __future__ import division
import numpy as np

from ..utils import solve_lsqr


def lawson_hanson(A, b, tol=1e-6, maxiter=int(1e5)):
    """Lawson-Hanson algorithm for computing the nonnegative least squares solution.

    The Lawson-Hanson algorithm modifies the active set by one element in each
    step. It combines forward selection with backwards steps that reduce the
    active set in case the estimate is not feasible in the active set.

    In general no upper bound on iterations. Worst case : 2^p (all possible
    subsets)

    .. note:: This algorithm can be drastically improved by keeping track
       of the candidates and/or using qr updating

    MATLAB package :: lsqnonneg


    Parameters
    ----------
    A

    b

    Examples
    --------

    References
    ----------
    [37]
    Cholesky updates : [47]
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

    # use boolean arrays
    active_set = np.ones(m, dtype=np.bool)
    passive_set = np.zeros(m, dtype=np.bool)

    while True:
        # compute negative gradient
        w = A.T.dot(b - A.dot(x))

        if not active_set and n_iter >= maxiter and w[active_set].max() > tol:
            # return solution
            break

        # update sets with minimum index of gradient (max of -grad)
        j = np.argmax(w[active_set])
        active_set[j] = 0
        passive_set[j] = 1

        while True:
            # make list so that we can index
            passive_list = list(passive_set)

            # compute least squares solution A[:, P]y = b
            y = solve_lsqr(A[:, passive_list], b)

            if y.min() <= tol:
                alpha = x[passive_list] / (x[passive_list] - y)

                # update feasibility vector
                x += alpha*(y - x)

                # move all zero indices of x from passive to active
                nonzero_x = x.nonzero()[0]
                active_set[nonzero_x] = 1
                passive_set[nonzero_x] = 0

            else:
                # update feasibility vector
                x = y
                break

        # update iter count
        n_iter += 1

    # return solution
    return x
