import numpy as np
from scipy.optimize import linprog

def radious_rep_x(n_rep_points, training, labels, x_mask, gamma, knn, current_column):
    
    M, N = training.shape
    nt_r_cls_1 = np.sum(labels)
    nt_r_cls_2 = N - nt_r_cls_1
    nc_1 = 0
    nc_2 = 0

    for i in range(n_rep_points):
        a = np.nonzero(x_mask[:, i] == 0)[0] # Identify
        if a.shape[0] < M:
            training_mask = x_mask[:, i].astype(bool)
            rep_points_p = training[training_mask, :]
            dist_rep = np.abs(np.sqrt(np.sum(( rep_points_p - rep_points_p[:, current_column][..., None] ) ** 2, 0)))
            ee_rep = np.msort(dist_rep)
            if labels[0, current_column] == 1:
                unique_values = np.unique(ee_rep)
                k = -1
                next_value = 1
                while next_value == 1:
                    k = k+1
                    r = unique_values[k]
                    f2 = (dist_rep <= r)
                    nc_ls_1_clst = np.sum(f2 & labels)-1
                    nc_ls_2_clst = np.sum(f2 & np.logical_not(labels))
                    if   gamma * (nc_ls_1_clst/(nt_r_cls_1-1)) < (nc_ls_2_clst/nt_r_cls_2):
                        next_value = 0
                        if (k) == 0:
                            r = unique_values[k]
                        else:
                            r = 0.5 * (unique_values[k-1] + unique_values[k])
                            nc_1 = nc_1 + 1
                        if r == 0:
                            r = 0.000001
                        r = 1 * r # pointless?
                        f2 = (dist_rep <= r)
                        nc_ls_1_clst = np.sum(f2 & labels)-1 # same as outside
                        nc_ls_2_clst = np.sum(f2 & np.logical_not(labels)) # same as outside
                        q = np.nonzero(f2 == 1)[0]
                        dr = 0
                        far = 0
                        for u in range(q.shape[0]):
                            quasi_test_p = rep_points_p[:, q[u]]

                            dist_quasi_test = np.absolute(np.sqrt(np.sum((rep_points_p - quasi_test_p[..., None]) ** 2, axis=0)))
                            min1 = np.sort(dist_quasi_test)
                            min_uniq = np.unique(min1)
                            m = 0
                            no_nereser = 0
                            while no_nereser < knn+1:
                                a1 = min_uniq[m]
                                NN = dist_quasi_test <= a1
                                no_nereser = np.sum(NN)
                                m = m + 1
                            no_nn_c1 = np.sum(NN & labels) # number of nearest neighbours
                            no_nn_c2 = np.sum(NN & np.logical_not(labels))
                            if labels[0, q[u]] == 1 and (no_nn_c1-1) > no_nn_c2:
                                dr = dr + 1
                            if labels[0, q[u]] == 0 and no_nn_c1 > (no_nn_c2-1):
                                far = far+1
###########################################################################################3
            if labels[0, current_column] == 0:
                unique_values = np.unique(ee_rep)
                k = -1
                next_value = 1

                while next_value == 1:
                    k = k + 1
                    r = unique_values[k]
                    f2 = (dist_rep <= r)
                    nc_ls_1_clst = np.sum(f2 & labels)
                    nc_ls_2_clst = np.sum(f2 & np.logical_not(labels))-1

                    if   gamma * (nc_ls_2_clst / (nt_r_cls_2-1)) < (nc_ls_1_clst / nt_r_cls_1):
                        next_value = 0
                        if (k) == 0: # We want to know if this is the first one
                            r = unique_values[k]
                        else:
                            r = 0.5 * (unique_values[k-1] + unique_values[k])
                            nc_2 = nc_2 + 1
                        if r == 0:
                            r = 0.000001
                            # r = 10**(-6)
                        r = 1*r
                        f2 = (dist_rep <= r)
                        nc_ls_1_clst = np.sum(f2 & labels)
                        nc_ls_2_clst = np.sum(f2 & np.logical_not(labels)) - 1               
                        q = np.where(f2 == 1)[0] # these are indices so should be 1 lower than our reference
                        dr = 0
                        far = 0

                        for u in range(q.shape[0]):

                            quasi_test_p = rep_points_p[:, q[u]]
                            dist_quasi_test = np.absolute(np.sqrt(np.sum((rep_points_p - quasi_test_p[..., None]) ** 2, axis=0)))
                            qausi_test_class = labels[0, q[u]]
                            min1 = np.sort(dist_quasi_test)
                            min_uniq = np.unique(min1)
                            m = 0
                            no_nereser = 0

                            while no_nereser < knn+1:
                                a1 = min_uniq[m]
                                NN = dist_quasi_test <= a1
                                no_nereser = np.sum(NN)
                                m = m + 1
                            no_nn_c1 = np.sum(NN & labels)
                            no_nn_c2 = np.sum(NN & np.logical_not(labels))
                            if qausi_test_class == 1 and (no_nn_c1-1) < no_nn_c2:
                                far = far + 1
                            if qausi_test_class == 0 and no_nn_c1 < (no_nn_c2 - 1):
                                dr = dr + 1
    return [r, dr, far]
