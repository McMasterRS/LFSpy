import numpy as np
from numpy import matlib
from scipy.optimize import fsolve, linprog

# Start Feature Selection Algorithm
def FES(neighbour_weight, M, N, patterns, targets, fstar, max_selected_features):
    
    # Preallocate arrays for parallel processing
    a = np.zeros((N, M)) # a is something
    b = np.zeros((N, M)) # b is something
    EpsilonMax = np.zeros((N, 1)) # EpsilonMax is something
    weight = np.zeros((N, N)) # weight is something

    # loop through every column (?) in the training data
    for i in range(N):

        cls1            = targets[0, i]             # Targets is a 1D array of 0 or 1
        temp_training   = np.delete(patterns, i, 1) # Get the pattern data without the i'th column
        temp_class      = np.delete(targets, i, 1)  # Get the target data without the i'th column
        Ncls1           = int(np.sum(temp_class))   # This is basically a count of all columns that are 1 in targets
        Ncls2           = int(temp_class.shape[1] - Ncls1) # This is basically a count of all columns that are 0 in targets
        A               = np.zeros((M, Ncls1))      # A will end up being all columns where targets is 1
        B               = np.zeros((M, Ncls2))      # B will end up being all columns where targets is 0

        k = 0 # index value / count for when temp class is 1
        m = 0 # index value / count for when temp class is 0

        # Initialize arrays for parallel processing
        f_cls1_temp = np.zeros((M, 1))
        f_cls2_temp = np.zeros((M, 1))
        f_Sparsity = np.zeros((M, 1))

        # for every column (?), fill the appropriate array A or B with the matching column data
        for j in range(temp_training.shape[1]):
            if temp_class[0, j] == 1:
                A[:, k] = temp_training[:, j]
                k = k + 1
            if temp_class[0, j] == 0:
                B[:, m] = temp_training[:, j]
                m = m + 1

        x = np.array([patterns[:, i]]).T # slice patterns at this column, and convert it to a vertical 2D array
        C = np.tile(x, (1, Ncls1))-A # does something
        D = np.tile(x, (1, Ncls2))-B # does something
        ro = C**2
        theta = D**2
        w11 = np.zeros((1, Ncls1))
        w22 = np.zeros((1, Ncls2))
        w = np.zeros((N ,N))
        w_ave = np.zeros((1,N))
        V = np.arange(1,N)
        V[i] = np.delete(V[i], i, 1)

        for p in range(0,N):
            fstar_slice = np.array([fstar[:, p]]).T
            fstar_tiled_ro = np.tile(fstar_slice,(1, Ncls1))
            fstar_tiled_alpha = np.tile(fstar_slice, (1, Ncls2))

            Dist_ro = np.sqrt(np.sum(fstar_tiled_ro*ro, axis=0))
            Dist_alpha = np.sqrt(np.sum(fstar_tiled_alpha*theta,axis=0))

            w11 = np.exp(-1*(Dist_ro-np.min(Dist_ro))**2/max_selected_features)
            w22 = np.exp(-1*(Dist_alpha-np.min(Dist_alpha))**2/max_selected_features)

            w[p, V] = np.concatenate((w11, w22)) # create a two dimensional array, with two rows, w11 and w22 and assign it to a particular spot in w
             
        w_ave = np.mean(w, axis=0)
        w_ave = np.delete(w_ave, i, 0) 
        w1 = w_ave[:Ncls1]
        w2 = w_ave[Ncls1:]
        w3 = 0
        normalized_w1 = w1/np.sum(w1)
        normalized_w2 = w2/np.sum(w2)
        wandering = 0
        w1 = (1-wandering)*normalized_w1+wandering*np.random.rand(1, w1.shape[0])
        w2 = (1-wandering)*normalized_w2+wandering*np.random.rand(1, w2.shape[0])   
        normalized_w1 = w1/np.sum(w1)
        normalized_w2 = w2/np.sum(w2)

        weight[:, i] = np.append(np.append(normalized_w1.T, normalized_w2.T), [0]) # we append the two and add a zero for padding
        w = weight[:, i]

        if cls1==1:
            for n in range(Ncls1):
                f_cls1_temp = f_cls1_temp + w[n] * ro[:, n]
            for n in range (Ncls2):
                f_cls2_temp = f_cls2_temp + w[n + Ncls1] * theta[:, n]

            f_Sparsity = w[-1] * np.ones((M, 1))
            f_temp = f_Sparsity + f_cls2_temp
            b[i, :] = f_temp.T / Ncls2
            a[i, :] = f_cls1_temp.T / Ncls1

        if cls1==0:
            for n in range(Ncls1):
                test = w[n] * ro[:, n]
                f_cls1_temp = f_cls1_temp + test[..., None]
            for n in range (Ncls2):
                test = w[n + Ncls1] * theta[:, n]
                f_cls2_temp = f_cls2_temp + test[..., None]

            f_Sparsity = w[-1] * np.ones((M, 1))
            f_temp = f_Sparsity + f_cls1_temp
            b[i, :] = f_temp.T / Ncls1
            a[i, :] = f_cls2_temp.T / Ncls2

        BB =  np.concatenate((np.ones((1, M)), -np.ones((1, M))), axis=0)
        b1 = np.array([[neighbour_weight, -1]]).T
        lb = np.zeros((M, 1))
        ub = np.ones((M, 1))

        linear_coefficients = -b[i, :].T
        inequality_constraints_a = BB
        inequality_constraints_b = b1
        bounds = (lb, ub)

        #         f,        A,   b,   Aeq,  beq, lb,      ub,       ?   options
        # linprog(-b(i,:)', BB,  b1,  [],   [],  lb,      ub,       [], opt_lin);
        #         c         A_ub b_ub A_eq, b_eq (bounds, bounds)   ?   ?

        res = linprog(
            linear_coefficients,
            A_ub=inequality_constraints_a,
            b_ub=inequality_constraints_b,
            bounds=(0, 1), # original code had homogenous arrays of 0 and 1 respectively
            method='interior-point'
        )

        if res.success != 1:
            print ('Not feasible')
        if res.success == 1:
            EpsilonMax[i] = -res.fun

        return [b, a, EpsilonMax]
