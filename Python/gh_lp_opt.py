import numpy as np
import scipy

# GH_LP_OPT linear programming optimization

def gh_lp_opt(NBeta, n, EpsilonMax, b, a, M, alpha):

    def linprog_callback(res):
        print(res)

    beta = 1/NBeta * n 
    epsilon = beta * EpsilonMax
    c = a
    A_ub = np.vstack((np.ones((1, M)), -np.ones((1, M)), -b))
    b_ub = np.vstack((alpha, -1, -epsilon))

    # linprog documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
    res = scipy.optimize.linprog(
        c, # Coefficients of the linear objective function to be minimized
        A_ub=A_ub, # The inequality constraint matrix. Each row of A_ub specifies the coefficients of a linear inequality constraint on x.
        b_ub=b_ub, # The inequality constraint vector. Each element represents an upper bound on the corresponding value of A_ub @ x.
        bounds=(0, 1), # A sequence of (min, max) pairs for each element in x, defining the minimum and maximum values of that decision variable.
        method='interior-point',
        options={
            'tol':0.000001,
            'maxiter': 200
        },
        # callback = linprog_callback
    )
    print(res.nit)
    return res.x[..., None], A_ub, b_ub, res.success
         # TT,               BB,   b1,   exit_flag