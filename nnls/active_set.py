"""
Active-set algorithms for computing the nonnegative least squares solution.
"""

# Authors: Joseph Knox <joseph.edward.knox@gmail.com>
# License: BSD 3

from __future__ import division
import numpy as np

from .utils import solve_lsqr


def block_pivoting(A, b, k_iter=10, tol=1e-8, max_iter=10000):
    """Block pivoting algorithm for computing the nonnegative least squares solution.

    Unlike in Lawson-Hanson and LARS, block pivoting changes multiple elements
    of the active set per iteration. Additionally, block pivoting is not
    guarenteed to terminate. Also, the iterates of the solution are not in
    general feasible. Thus, block pivoting performs well for smaller scale,
    well conditioned problems.

    Will raise warning if A is not full rank.

    Parameters
    ----------
    A

    b

    Examples
    --------

    References
    ----------
    [37]
    [57]
    """
    m = A.shape[1]

    # initialize
    n_iter = 0
    k = k_iter
    x = np.zeros(m)
    y = -A.T.dot(b)

    active_set = np.ones(m, dtype=np.bool)
    n_inf = m + 1

    while n_inf > 0 and n_iter < max_iter:

        # compute gradient
        y = A.T.dot(A.dot(x) - b)

        # get infeasible sets
        x_inf = x[active_set] < 0
        y_inf = y[~active_set] < 0

        sum_inf = x_inf.sum() + y_inf.sum()

        if sum_inf < n_inf:
            # n_infeasible reduced: update according to Kostreva's
            k = k_iter
            active_set[x_inf] = 0
            active_set[y_inf] = 1

        else:
            if k > 1:
                # update according to Kostreva's
                k -= 1
                active_set[x_inf] = 0
                active_set[y_inf] = 1
            else:
                # use Murty's to find x st. |x_inf & y_inf| < n_inf
                r = np.nonzero(x_inf & y_inf)[0][-1]

                # is r in x or y
                in_x = x_inf[r]

                # reset sets
                x_inf = np.zeros(m, dtype=np.bool)
                y_inf = np.zeros(m, dtype=np.bool)

                if in_x:
                    x_inf[r] = 1
                else:
                    y_inf[r] = 1

        # update x
        x = np.zeros(m)
        x[active_set] = solve_lsqr(A[:, active_set], b)


        # update iter count
        n_iter += 1

    # return solution
    return x


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
        # if active set emptied: ls solution == nnls solution

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


#def nonnegative_lasso_lars(A, b, tol=1e-8, max_iter=10000):
#    """Least Angle Regressions with nonnegative lasso constraint.
#
#    It can be shown that the solution to the nonnegative lasso variation of
#    LARS results in the nonnegative least squares solution
#
#    Parameters
#    ----------
#    A
#
#    b
#
#    Examples
#    --------
#
#    References
#    ----------
#    [37]
#    [26]
#    """
#    m = A.shape[1]
#
#    # initialize
#    n_iter = 0
#    x = np.zeros(m)
#    active_set = np.ones(m, dtype=np.bool)
#
#    while n_iter < max_iter and active_set.any():
#        # if active set emptied: ls solution == nnls solution
#
#        # compute negative gradient
#        w = A.T.dot(b - A.dot(x))
#
#        if w[active_set].max() < tol:
#            # satisfication of KKT condistion; return solution
#            break
