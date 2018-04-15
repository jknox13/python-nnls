"""
Active-set algorithms for computing the nonnegative least squares solution.
"""

# Authors: Joseph Knox <joseph.edward.knox@gmail.com>
# License: BSD 3

from __future__ import division
import warnings
import numpy as np

from .utils import matrix_rank, solve_lsqr


def block_pivoting(A, b, max_k_iter=5, tol=1e-8, max_iter=10000):
    """Block pivoting algorithm for computing the nonnegative least squares solution.

    Unlike in Lawson-Hanson and LARS, block pivoting changes multiple elements
    of the active set per iteration. Additionally, block pivoting is not
    guarenteed to terminate. Also, the iterates of the solution are not in
    general feasible. Thus, block pivoting performs well for smaller scale,
    well conditioned problems.


    Parameters
    ----------
    A

    b

    max_k_iter:
        backup rule


    References
    ----------
    [37]
    [57]
    """
    # method only valid for full rank design matrices
    if matrix_rank(A) < A.shape[1]:
        warnings.warn('A does not have full column rank! block_pivoting is '
                      'only reliable for matrices with full column rank')

    m = A.shape[1]

    # initialize
    n_iter = 0
    k = max_k_iter
    n_infeasible = m + 1
    active_set = np.ones(m, dtype=np.bool)
    x = np.zeros(m)
    y = -A.T.dot(b)

    while True:
        if n_iter >= max_iter:
            warnings.warn('Max iterations reached: did not converge')

        # get infeasible set
        infeasible = np.append((x < 0).nonzero()[0], (y < 0).nonzero()[0])

        if infeasible.size == 0:
            # solution feasible; return result
            break

        if k >= 1:
            # update according to Kostreva's
            invert = infeasible

            if infeasible.size < n_infeasible:
                # update previous infesible cardinality
                n_infeasible = infeasible.size
                k = max_k_iter

            else:
                # decrease the remaining Kostreva iterations
                k -= 1

        else:
            # backtrack using only a single pivot (Murty's)
            invert = infeasible.max()

        # update active set
        active_set[invert] = np.bitwise_not(active_set[invert])

        # update complementary solution
        x = solve_lsqr(A[:, ~active_set], b)
        y = A[:, active_set].T.dot(A[:, ~active_set].dot(x) - b)

        # update iter count
        n_iter += 1

    # return complementary solution
    solution = np.zeros(m)
    solution[active_set] = y
    solution[~active_set] = x

    return solution


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

    while active_set.any():
        # if active set emptied: ls solution == nnls solution

        if n_iter >= max_iter:
            warnings.warn('Max iterations reached: did not converge')

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
