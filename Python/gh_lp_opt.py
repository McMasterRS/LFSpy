import numpy as np
import scipy

def gh_lp_opt(NBeta, n, EpsilonMax, b, a, M, alpha):

    beta = 1/NBeta * n
    epsilon = beta * EpsilonMax
    b1 = np.array((alpha, -1, -epsilon)).T
    BB = np.vstack( (np.ones((1, M)), -np.ones((1, M))) )
    BB = np.vstack( (BB, np.array(-b) ))
    lb = np.zeros((M, 1))
    ub = np.ones((M, 1))

    # complete outputs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
    res = scipy.optimize.linprog(a.conj().T,
        A_ub = BB,
        b_ub = b1,
        bounds = (0, 1),
        method='interior-point'
    )
    return res.x[..., None], BB, b1, lb, ub, res.success