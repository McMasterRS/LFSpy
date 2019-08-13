# class stub for interfacing with scikit learn pipeline.
# Fit, transform, and fit_transform methods are required

# gamma:   impurity level (default: 0.2)
# tau:     number of iterations (default: 2)
# sigma:   controls neighboring samples weighting (default: 1)
# alpha:   maximum number of selected feature for each representative point
# NBeta:   numer of distinct \beta (default: 20)
# NRRP:    number of iterations for randomized rounding process (defaul 2000)

import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog


class LFS():

    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self):
        pass

    def fit_transform(self):
        pass

###########################################

    def classification(self):
        pass
    
    def feature_evaluation(self, neighbour_weight, n_beta, epsilon_max, b, a, train, train_labels, impurity_level, nrrp, knn):

        M = b.shape[1]
        N = train.shape[1]

        for i in range(N):
            for j in range(n_beta):
                beta = 1/n_beta  * (j+1)
                epsilon = beta * epsilon_max[i, 0]
                a_ub = np.vstack((np.ones((1, M)), -np.ones((1, M)), -b[i, :]))
                b_ub = np.vstack((neighbour_weight, -1, -epsilon))

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
                success = res.success
                #### snapping ####
                a_slice = a[i, :]
                b_slice = b[i, :]
                r = loadmat('./data/r')['r']
                unq = np.unique((r <= x_values).T, axis=0).T
                n_unq = unq.shape[1]
                
                x_binary_temp = np.zeros((M, 1))
                wit_dist = np.inf * np.ones((1, n_unq)) # within distance
                btw_dist = -np.inf * np.ones((1, n_unq)) # between distance
                dr = np.zeros((1, n_unq))
                far = np.zeros((1, n_unq))
                feasib = np.zeros((1, n_unq))

                for k in range(n_unq):
                    x_binary_temp[unq[:, k]] = 1
                    if np.sum(a_ub @ x_binary_temp > b_ub) == 0:  # if atleast one feature is active and no more than maxNoFea
                        feasib[0, k] = 1
                        wit_dist[0, k] = a_slice @ x_binary_temp
                        btw_dist[0, k] = b_slice @ x_binary_temp
                        [radious[0, k], dr[0, k], far[0, k]] = self.radious_rep_x(N, 1, train, train_labels, x_binary_temp, 0.2, knn, i)
                #### /snapping ####

    # n representative points?
    def radious_rep_x(self, N, n_rep_points, train, train_labels, x_values, gamma, knn, current_column):
        n_cls_1 = np.sum(train_labels) # n_tr_cls_1 number of class 1
        n_cls_2 = N - n_cls_1 # number of class 2
        M = train.shape[0]
        nc_1 = 0
        nc_2 = 0
        for i in range(n_rep_points):
            a = np.nonzero(x_values[:,i] == 0)[0]
            if a.shape[0] < M:
                train_mask = x_values[:, i].astype(bool) # Get all of the rows at the i'th column of T
                rep_points_p = train[train_mask, :]
                dist_rep = np.abs(np.sqrt(np.sum(( (rep_points_p - rep_points_p[:, current_column][..., None]).T ) ** 2, 0)))
                ee_rep = np.msort(dist_rep)
                unq = np.unique(ee_rep)
                k = 0
                continue_processing = True
                while continue_processing:
                    r = unq[k]
                    f2 = (dist_rep <= r) # where distance from representative point is less than the something (not random)
                    n_cls_1_clst = np.sum(f2 & train_labels) # number of closest class 1 perhaps
                    n_cls_2_clst = np.sum(f2 & np.logical_not(train_labels)) # number of closest class 2 perhaps
                    if train_labels[0, current_column] == 1:
                        n_cls_2_clst = n_cls_2_clst - 1
                        if   gamma * (n_cls_1_clst/(n_cls_1-1)) < (n_cls_2_clst/n_cls_2):
                            continue_processing = False
                            if k == 0:
                                r = unq[k]
                            else:
                                r = 0.5 * (unq[k-1] + unq[k])
                                nc_1 = nc_1 + 1
                            if r == 0:
                                r = 0.000001
                            r = 1 * r
                            f2 = (dist_rep <= r)
                            n_cls_1_clst = np.sum(f2 & train_labels)-1
                            n_cls_2_clst = np.sum(f2 & np.logical_not(train_labels))
                            q = np.where(f2 == 1)[0]
                            f2[(f2==1)].shape[0]
                            dr = 0
                            far = 0
                            for u in range(q.shape[0]):
                                quasi_test_p = rep_points_p[:, q[u]]

                                dist_quasi_test = np.absolute(np.sqrt(np.sum((rep_points_p - quasi_test_p)) ** 2, axis=0))

                                min1 = np.sort(dist_quasi_test[...,None], axis=1)
                                min_uniq = np.unique(min1)
                                m = -1
                                no_nereser = 0
                                while no_nereser < knn+1:
                                    m = m + 1
                                    a1 = min_uniq[m]
                                    NN = dist_quasi_test <= a1
                                    no_nereser = np.sum(NN)
                                no_nn_c1 = np.sum(NN & targets) # number of nearest neighbours
                                no_nn_c2 = np.sum(NN & np.logical_not(targets))
                                if targets[0, q[u]] == 1 and (no_nn_c1-1) > no_nn_c2:
                                    dr = dr + 1
                                if targets[0, q[u]] == 0 and no_nn_c1 > (no_nn_c2-1):
                                    far = far+1




    def accuracy(self):
        pass
    
    # Feature Selection (train == patterns, train_labels == targets)
    def feature_selection(self, train, train_labels, fstar, max_selected_features, neighbour_weight):

        WANDERING = 0
        M, N = train.shape

        a = np.zeros((N, M)) # a is something
        b = np.zeros((N, M)) # b is something
        epsilon_max = np.zeros((N, 1)) # EpsilonMax is something

        mask = np.ones((N), dtype=bool)
        for i in range(N):
            mask[i] = False

            if(i - 1 > 0):
                mask[i-1] = True

            cls1 = train_labels[0, i] #
            temp_training = train[:, mask] # training data for all but ith column
            temp_labels = train_labels[:, mask] # training data for all but ith column
            n_cls1 = int(np.sum(temp_labels)) # number of instances of class 1
            n_cls2 = int(temp_labels.shape[1] - n_cls1) # number of isntances of class 0
            V = np.arange(0, N)[mask]
            class_mask = temp_labels[0].astype(bool)

            ro = (-temp_training[:, class_mask] + train[:, i][..., None])**2
            theta = (-temp_training[:, ~class_mask] + train[:, i][..., None])**2

            w = np.zeros((N, N))
            w_ave = np.zeros((1,N))
            for j in range(0, N):
                fstar_slice = fstar[:, j][..., None]
                dist_ro = np.sqrt(np.sum(fstar_slice*ro, axis=0))
                dist_alpha = np.sqrt(np.sum(fstar_slice*theta, axis=0))
                w[j, V] = np.concatenate(( 
                    np.exp(-dist_ro - np.min(dist_ro) ** 2 / max_selected_features),
                    np.exp(-dist_alpha - np.min(dist_alpha) ** 2 / max_selected_features)
                ))
            
            w_ave = np.mean(w, axis=0)[mask]
            w1 = w_ave[:n_cls1]
            w2 = w_ave[n_cls1:]
            normalized_w1 = w1/np.sum(w1)
            normalized_w2 = w2/np.sum(w2)
            w1 = (1-WANDERING) * normalized_w1 + WANDERING * np.random.rand(1, w1.shape[0])
            w2 = (1-WANDERING) * normalized_w2 + WANDERING * np.random.rand(1, w2.shape[0])   
            normalized_w1 = w1/np.sum(w1)
            normalized_w2 = w2/np.sum(w2)
            w = np.concatenate((normalized_w1.T, normalized_w2.T, np.zeros((1, 1))), axis=0) # we append the two and add a zero for padding
            f_sparsity = w[-1] * np.ones((M, 1))
    
            f_cls1_temp = np.zeros((M, 1))
            for j in range(n_cls1):
                f_cls1_temp = f_cls1_temp + (w[j] * ro[:, j])[..., None]

            f_cls2_temp = np.zeros((M, 1))
            for j in range (n_cls2):
                f_cls2_temp = f_cls2_temp + (w[j + n_cls1] * theta[:, j])[..., None]
    
            if cls1==1:
                b[i, :] = (f_sparsity + f_cls2_temp).T / n_cls2
                a[i, :] = f_cls1_temp.T / n_cls1

            elif cls1==0:
                b[i, :] = (f_sparsity + f_cls1_temp).T / n_cls1
                a[i, :] = f_cls2_temp.T / n_cls2

            c = -b[i, :].T
            A_ub = np.concatenate((np.ones((1, M)), -np.ones((1, M))), axis=0)
            b_ub = np.array([[neighbour_weight, -1]]).T

            res = linprog(
                c,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=(0, 1),
                method='interior-point',
                options={
                    'tol':0.000001,
                    'maxiter': 200
                }
            )

            if res.success:
                epsilon_max[i] = -res.fun
            else:
                print ('Not feasible')

        return [b, a, epsilon_max]

mat = loadmat('matlab_Data')

train               = mat['Train']
train_labels        = mat['TrainLables']
fstar               = np.zeros(train.shape)
neighbour_weight    = 19
max_selected_features = 1
n_beta              = 20     # number of distinct beta (BETA, β)
nrrp                = 2000      # Number randomized rounding permutations
knn                 = 1
impurity_level = 0.2
lfs = LFS()

b, a, epsilon_max = lfs.feature_selection(train, train_labels, fstar, max_selected_features, neighbour_weight)
tb_temp, tr_temp, b_ratio, feasib, radious = lfs.feature_evaluation(neighbour_weight, n_beta, epsilon_max, b, a, train, train_labels, impurity_level, nrrp, knn)