import numpy as np
import scipy

def gh_lp_opt(NBeta, n, EpsilonMax, b, a, M, alpha):

    beta = 1/NBeta * n
    epsilon = beta * EpsilonMax
    b1 = np.array((alpha, -1, -epsilon)).T
    BB = np.vstack((np.ones((1, M)), -np.ones((1, M)), -b))
    test = np.sum(-b)

    # linprog documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
    res = scipy.optimize.linprog(a[..., None],
        A_ub = BB, # equality constraint matrix coefficients
        b_ub = b1, # equality constraint vectors
        bounds = (0.0, 1.0),
        method='interior-point',
        options={
            'tol':0.000001,
            'maxiter': 200
        }
    )
    return res.x[..., None], BB, b1, res.success