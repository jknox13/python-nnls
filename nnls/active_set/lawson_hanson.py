"""
Lawson-Hanson algorithm for computing the nonnegative least squares solution.
"""

# Authors: Joseph Knox <joseph.edward.knox@gmail.com>
# License: BSD 3

from __future__ import division
import numpy as np

from ..utils import solve_lsqr


def lawson_hanson(A, b, tol=1e-8, max_iter=10000):
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
    [47]
    """
    m = A.shape[1]

    # initialize
    n_iter = 0
    x = np.zeros(m)
    active_set = np.ones(m, dtype=np.bool)

    while n_iter < max_iter and active_set.any():
        # if active set emptied => ls solution == nnls solution

        # compute negative gradient
        w = A.T.dot(b - A.dot(x))

        if w[active_set].max() < tol:
            # satisfication of KKT condistion; return solution
            break

        # update sets with minimum index of gradient (max of -grad)
        j = np.argmax(w[active_set])
        active_set[active_set.nonzero()[0][j]] = 0

        while True:
            # get updated passive_set
            passive_set = ~active_set

            # compute least squares solution A[:, P]y = b
            y = np.zeros(m)
            y[passive_set] = solve_lsqr(A[:, passive_set], b)

            infeasible = (y < tol) & passive_set
            if infeasible.any():

                # find points along line segment xy to backtrack
                alpha = x[infeasible] / (x[infeasible] - y[infeasible])

                # backtrack feasibility vector
                x += alpha.min()*(y - x)

                # move all zero* indices of x from passive to active
                active_set[(np.abs(x) < tol) & passive_set] = 1

            else:
                # update feasibility vector
                x = y
                break

        # update iter count
        n_iter += 1

    # return solution
    return x
