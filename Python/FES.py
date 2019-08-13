import numpy as np
from numpy import matlib
from scipy.optimize import fsolve, linprog
from scipy.io import loadmat

# Start Feature Selection Algorithm
def FES(neighbour_weight, M, N, patterns, targets, fstar, max_selected_features):
    
    # Preallocate arrays
    a = np.zeros((N, M)) # a is something
    b = np.zeros((N, M)) # b is something
    EpsilonMax = np.zeros((N, 1)) # EpsilonMax is something
    weight = np.zeros((N, N)) # weight is something

    # loop through every column (?) in the training data
    for i in range(N):
        i_mask          = np.ones((N), dtype=bool)
        i_mask[i]       = False


        cls1            = targets[0, i]             # 
        temp_training   = patterns[:, i_mask]       # Patterns with out the i'th column
        temp_class      = targets[:, i_mask]        # Slice out the i'th column from targets
        Ncls1           = int(np.sum(temp_class))   # Count the number of 1s in targets
        Ncls2           = int(temp_class.shape[1] - Ncls1) # count the number of 0s in targets
        V               = np.arange(0, N)[i_mask]
        # temp_training   = np.delete(patterns, i, 1) # Get the pattern data without the i'th column
        # temp_class      = np.delete(targets, i, 1)  # Get the target data without the i'th column
        A               = np.zeros((M, Ncls1))      # A will end up being all columns where targets is 1
        B               = np.zeros((M, Ncls2))      # B will end up being all columns where targets is 0
        w11             = np.zeros((1, Ncls1))
        w22             = np.zeros((1, Ncls2))
        w               = np.zeros((N, N))
        w_ave           = np.zeros((1,N))
        f_cls1_temp     = np.zeros((M, 1))
        f_cls2_temp     = np.zeros((M, 1))
        f_Sparsity      = np.zeros((M, 1))

        k = 0 # index value / count for when temp class is 1
        m = 0 # index value / count for when temp class is 0

        # for every column (?), fill the appropriate array A or B with the matching column data
        # for j in range(temp_training.shape[1]):
        #     if temp_class[0, j] == 1:
        #         A[:, k] = temp_training[:, j]
        #         k = k + 1
        #     if temp_class[0, j] == 0:
        #         B[:, m] = temp_training[:, j]
        #         m = m + 1

        # x = np.array([patterns[:, i]]).T # slice patterns at this column, and convert it to a vertical 2D array
        # C = np.tile(x, (1, Ncls1))-A # does something
        # D = np.tile(x, (1, Ncls2))-B # does something
        class_mask = temp_class[0].astype(bool)
        A = temp_training[:, class_mask]    # All of the rows where the class is 1
        B = temp_training[:, ~class_mask]   # All of the rows where the class is 0
        C = -A + patterns[:, i][..., None]  # Subtract the i'th column of patterns from training data class 1
        D = -B + patterns[:, i][..., None]  # Subtract the i'th column of patterns from training data class 0

        ro = C**2
        theta = D**2

        for p in range(0,N):
            # fstar_tiled_ro = np.tile(fstar_slice,(1, Ncls1))
            # fstar_tiled_alpha = np.tile(fstar_slice, (1, Ncls2))

            fstar_slice = fstar[:, p][..., None]
            Dist_ro = np.sqrt(np.sum(fstar_slice*ro, axis=0))
            Dist_alpha = np.sqrt(np.sum(fstar_slice*theta, axis=0))

            w11 = np.exp(-Dist_ro - np.min(Dist_ro) ** 2 / max_selected_features)
            w22 = np.exp(-Dist_alpha - np.min(Dist_alpha) ** 2 / max_selected_features)

            w[p, V] = np.concatenate((w11, w22)) # create a two dimensional array, with two rows, w11 and w22 and assign it to a particular spot in w
             
        w_ave = np.mean(w, axis=0)
        w_ave = np.delete(w_ave, i) 
        w1 = w_ave[:Ncls1]
        w2 = w_ave[Ncls1:]
        w3 = 0
        normalized_w1 = w1/np.sum(w1)
        normalized_w2 = w2/np.sum(w2)
        wandering = 0
        w1 = (1-wandering) * normalized_w1 + wandering * np.random.rand(1, w1.shape[0])
        w2 = (1-wandering) * normalized_w2+wandering * np.random.rand(1, w2.shape[0])   
        normalized_w1 = w1/np.sum(w1)
        normalized_w2 = w2/np.sum(w2)

        weight[:, i] = np.append(np.append(normalized_w1.T, normalized_w2.T), [0]) # we append the two and add a zero for padding
        w = weight[:, i]
        f_Sparsity = w[-1] * np.ones((M, 1))

        for n in range(Ncls1):
            f_cls1_temp = f_cls1_temp + (w[n] * ro[:, n])[..., None]

        for n in range (Ncls2):
            f_cls2_temp = f_cls2_temp + (w[n + Ncls1] * theta[:, n])[..., None]

        if cls1==1:
            f_temp = f_Sparsity + f_cls2_temp
            b[i, :] = f_temp.T / Ncls2
            a[i, :] = f_cls1_temp.T / Ncls1

        if cls1==0:
            f_temp = f_Sparsity + f_cls1_temp
            b[i, :] = f_temp.T / Ncls1
            a[i, :] = f_cls2_temp.T / Ncls2

        BB = np.concatenate((np.ones((1, M)), -np.ones((1, M))), axis=0)
        b1 = np.array([[neighbour_weight, -1]]).T
        lb = np.zeros((M, 1))
        ub = np.ones((M, 1))

        linear_coefficients = -b[i, :].T
        inequality_constraints_a = BB
        inequality_constraints_b = b1
        bounds = (lb, ub)

        res = linprog(
            linear_coefficients,
            A_ub = inequality_constraints_a,
            b_ub = inequality_constraints_b,
            bounds = (0, 1), # original code had homogenous arrays of 0 and 1 respectively
            method = 'interior-point',
            options = {
                'tol':0.000001,
                'maxiter': 200
            }
        )

        if not res.success:
            print ('Not feasible')
        if res.success:
            EpsilonMax[i] = -res.fun
    return [b, a, EpsilonMax]