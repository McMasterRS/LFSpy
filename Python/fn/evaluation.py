import numpy as np
from scipy.optimize import linprog
from scipy.io import loadmat
from fn.radious_rep_x import radious_rep_x

# gamma:   impurity level (default: 0.2)
# tau:     number of iterations (default: 2)
# sigma:   controls neighboring samples weighting (default: 1)
# alpha:   maximum number of selected feature for each representative point
# NBeta:   numer of distinct \beta (default: 20)
# NRRP:    number of iterations for randomized rounding process (defaul 2000)

def evaluation(training_data, training_labels, a, b, epsilon_max, params):

    alpha = params['alpha']
    gamma = params['gamma']
    nrrp = params['nrrp']
    knn = params['knn']
    n_beta = params['n_beta']

    M = b.shape[1]
    N = training_data.shape[1]
    n_class_1 = np.sum(training_labels)
    n_class_2 = np.sum(np.logical_not(training_labels)) # where this is 0
    
    TRTemp = np.zeros((M, N, n_beta))
    TBTemp = np.zeros((M, N, n_beta))
    Bratio = -2*np.ones((N, n_beta))
    feasib = np.zeros((N, n_beta))
    radious = np.zeros((N, n_beta))

    for i in range(N):
        for j in range(n_beta):
            #### lp_opt ####
            beta = 1/n_beta  * (j+1)
            epsilon = beta * epsilon_max[i, 0]
            a_ub = np.vstack((np.ones((1, M)), -np.ones((1, M)), -b[i, :]))
            b_ub = np.vstack((alpha, -1, -epsilon))

            # linprog documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
            res = linprog(
                a[i, :], # Coefficients of the linear objective function to be minimized
                A_ub=a_ub, # The inequality constraint matrix. Each row of A_ub specifies the coefficients of a linear inequality constraint on x.
                b_ub=b_ub, # The inequality constraint vector. Each element represents an upper bound on the corresponding value of A_ub @ x.
                bounds=(0, 1), # A sequence of (min, max) pairs for each element in x, defining the minimum and maximum values of that decision variable.
                method='interior-point',
                options={
                    'tol':0.000001,
                    'maxiter': 200
                }
            )
            x_values = res.x[..., None]
            #### /LP_OPT ####
            #### snapping ####
            if res.success == 1: # if we converged upon a point successfully
                a_slice = a[i, :]
                b_slice = b[i, :]
                r = loadmat('./data/r')['r'] # this should be replaced by the correct randomization algorithm
                unq = np.unique((r <= x_values).T, axis=0).T
                n_unq = unq.shape[1] # number of unique values

                x_binary_temp = np.zeros((M, 1)) # this is some sort of mask
                radious_temp = np.zeros((1, n_unq))
                feasib_temp = np.zeros((1, n_unq))
                dr = np.zeros((1, n_unq))
                far = np.zeros((1, n_unq))
                wit_dist = np.inf * np.ones((1, n_unq)) # within distance
                btw_dist = -np.inf * np.ones((1, n_unq)) # between distance

                for k in range(n_unq):
                    x_binary_temp[unq[:, k]] = 1
                    x_binary_temp[~unq[:, k]] = 0
                    if np.sum(a_ub @ x_binary_temp > b_ub) == 0:  # if atleast one feature is active and no more than maxNoFeatures
                        feasib_temp[0, k] = 1
                        wit_dist[0, k] = a_slice @ x_binary_temp
                        btw_dist[0, k] = b_slice @ x_binary_temp
                        [radious_temp[0, k], dr[0, k], far[0, k]] = radious_rep_x(1, training_data, training_labels, x_binary_temp, 0.2, knn, i)

                eval_criteria = [
                    dr/n_class_1-far/n_class_2,
                    dr/n_class_2-far/n_class_1
                ]
                b1 = np.argmin(wit_dist) # find the shortest within distance
                TT_binary = unq[:, b1]
                feasib[i, j] = feasib_temp[0, b1]
                radious[i, j] = radious_temp[0, b1]
                Bratio[i, j] = eval_criteria[training_labels[0, i]][0, b1]

                if feasib[i, j] == 1:
                    TBTemp[:, i, j] = x_binary_temp[:, 0]
                    TRTemp[:, i, j] = x_values[:, 0]

    return TBTemp, TRTemp, Bratio, feasib, radious