import math
import numpy as np

def radious_rep_x(N, no_rep_points, patterns, targets, t, gamma, y, knn):
    """ """    
    nt_r_cls_1 = np.sum(targets)
    nt_r_cls_2 = N - nt_r_cls_1
    m = patterns.shape[0]
    nc_1 = 0
    nc_2 = 0

    for i in range(no_rep_points):
        a = np.nonzero(t[:, i] == 0)[0] # Identify
        if a.shape[0] < m:
            patterns_mask = t[:, i].astype(bool) # Get all of the rows at the i'th column of T
            rep_points_p = patterns[patterns_mask, :]
            dist_rep = np.abs(
                np.sqrt(
                    np.sum(
                        ( rep_points_p - np.tile(rep_points_p[:, y], (N, 1)).T ) ** 2
                    , 0)
                )
            )
            ee_rep = np.msort(dist_rep)
            if targets[0, y] == 1:
                unique_values = np.unique(ee_rep)
                k = -1
                next_value = 1
                while next_value == 1:
                    k = k+1
                    r = unique_values[k]
                    f2 = (dist_rep <= r)
                    nc_ls_1_clst = np.sum(f2 & targets)-1
                    nc_ls_2_clst = np.sum(f2 & np.logical_not(targets))
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
                        nc_ls_1_clst = np.sum(f2 & targets)-1 # same as outside, nope f2 is different
                        nc_ls_2_clst = np.sum(f2 & np.logical_not(targets)) # same as outside
                        q = np.nonzero(f2 == 1)[0]
                        dr = 0
                        far = 0
                        for u in range(q.shape[0]):
                            quasi_test_p = rep_points_p[:, q[u]]

                            dist_quasi_test = np.absolute(
                                np.sqrt(
                                    np.sum(
                                        (
                                            rep_points_p - np.tile(quasi_test_p[..., None], (1, N))
                                        ) ** 2, axis=0
                                    )
                                )
                            )

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
###########################################################################################3
            if targets[0, y] == 0:
                unique_values = np.unique(ee_rep)
                k = -1
                next_value = 1

                while next_value == 1:
                    k = k + 1
                    r = unique_values[k]
                    f2 = (dist_rep <= r)
                    nc_ls_1_clst = np.sum(f2 & targets)
                    nc_ls_2_clst = np.sum(f2 & np.logical_not(targets))-1

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
                        nc_ls_1_clst = np.sum(f2 & targets)
                        nc_ls_2_clst = np.sum(f2 & np.logical_not(targets)) - 1               
                        q = np.where(f2 == 1)[0] # these are indices so should be 1 lower than our reference
                        dr = 0
                        far = 0

                        for u in range(q.shape[0]):

                            quasi_test_p = rep_points_p[:, q[u]]
                            _pow = np.power((rep_points_p - np.tile(quasi_test_p, (N, 1)).T), 2)
                            _sum = np.sum(_pow, axis=0)
                            _sqrt = np.sqrt(_sum)
                            dist_quasi_test = np.abs(_sqrt)[None, ...] # B
                            qausi_test_class = targets[0, q[u]]
                            min1 = np.sort(dist_quasi_test, axis=1)
                            min_uniq = np.unique(min1)
                            m = -1
                            no_nereser = 0

                            while no_nereser < knn+1:
                                m = m + 1
                                a1 = min_uniq[m]
                                NN = dist_quasi_test <= a1
                                no_nereser = np.sum(NN)
                            no_nn_c1 = np.sum(NN & targets)
                            no_nn_c2 = np.sum(NN & np.logical_not(targets))
                            if qausi_test_class == 1 and (no_nn_c1-1) < no_nn_c2:
                                far = far + 1
                            if qausi_test_class == 0 and no_nn_c1 < (no_nn_c2 - 1):
                                dr = dr + 1
    return [r, dr, far]