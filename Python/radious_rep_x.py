import math
import numpy as np

def radious_rep_x(N, no_rep_points, patterns, targets, t, gamma, y, knn):
    """ """
    print('ENTERING: Radious rep x')

    # N == 100, no_rep_points == 1, patterns == [160x100], targets == [1x100], t === [160x1], gamma == 0.2, y == 1, knn == 1
    print("100 1 (160, 100) (1, 100) (160, 1) 0.2 0 1")
    print(N, no_rep_points, patterns.shape, targets.shape, t.shape, gamma, y, knn)
    
    nt_r_cls_1 = np.sum(targets)
    nt_r_cls_2 = N - nt_r_cls_1
    m = patterns.shape[0]
    nc_1 = 0
    nc_2 = 0

    # nt_r_cls_1 == 43, nt_r_cls_2 == 57, m == 160
    print("43 57.0 160")
    print(nt_r_cls_1, nt_r_cls_2, m)

    for i in range(no_rep_points):
        a = np.nonzero(t[:, i][..., None] == 0)[0][..., None]

        print("(158, 1), [11], [122]")
        print(a.shape, a[10], a[121])

        if a.shape[1] < m:

            rep_points_p = patterns[(t.T[0] == 1), :]

            print("% (2, 100) [ 2.1520 -0.999 ]")
            print("#", rep_points_p.shape, rep_points_p[:, 0])

            dist_rep = np.abs( np.sqrt( np.sum(( rep_points_p - np.tile( rep_points_p[:, y], (N, 1)).T) ** 2, 0)))
            ee_rep = np.msort(dist_rep)
            print("(1, 100) [0	2.02902372611516	2.19949289620160	2.11578128365065] (1, 100) 0	0.0501671240515843	0.0652832179008543	0.229052705182234")
            print(dist_rep[None, ...].shape, dist_rep[:4], ee_rep[None, ...].shape, ee_rep[:4])
            print("asdfasdf", targets[0, y])
            if targets[0, y] == 1:
                unique_values = np.unique(ee_rep)
                k = 0
                next_value = 1
                while next_value == 1:
                    k = k+1
                    r = unique_values[k]
                    # f1 = (dist_rep == r) # UNUSED VARIABLE
                    # nc_ls_1_r = sum(F1 and targets) # UNUSED VARIABLE
                    # nc_ls_2_r = sum(F1 and not(targets)) # UNUSED VARIABLE
                    f2 = (dist_rep <= r)
                    nc_ls_1_clst = np.sum(f2 & targets)-1
                    nc_ls_2_clst = np.sum(f2 & ~targets)
                    if   gamma * (nc_ls_1_clst/(nt_r_cls_1-1)) < (nc_ls_2_clst/nt_r_cls_2):
                        next_value = 0
                        if (k-1) == 0:
                            r = unique_values[k]
                        else:
                            r = 0.5 * (unique_values[k-1] + unique_values[k])
                            nc_1 = nc_1 + 1
                        if r == 0:
                            r = 10 ** -6
                        r = 1 * r
                        f2 = (dist_rep <= r)
                        nc_ls_1_clst = np.sum(f2 & targets)-1
                        nc_ls_2_clst = np.sum(f2 & ~targets)
                        q = np.where(f2 == 1)[1]
                        dr = 0
                        far = 0
                        for u in range(q.shape):
                            quasi_test_p = rep_points_p[:, q(u)]
                            dist_quasi_test = abs(math.sqrt(sum((rep_points_p - np.tile(quasi_test_p, (1, N))) ** 2, 1)))
                            min1 = np.sort(dist_quasi_test, axis=2)
                            min_uniq = np.unique(min1)
                            m = 0
                            no_nereser = 0
                            while no_nereser < knn+1:
                                m = m+1
                                a1 = min_uniq[m]
                                NN = dist_quasi_test <= a1
                                no_nereser = sum(NN)
                            no_nn_c1 = sum(NN and targets)
                            no_nn_c2 = sum(NN and not(targets))
                            if targets[1, q[u]] == 1 and (no_nn_c1-1) > no_nn_c2:
                                dr = dr + 1
                            if targets[1, q[u]] == 0 and no_nn_c1 > (no_nn_c2-1):
                                FAR = FAR+1

            if targets[0, y] == 0:
                unique_values = np.unique(ee_rep)

                print("(1, 100)")
                print(unique_values)

                k = -1
                next_value = 1

                while next_value == 1:
                    print(k)
                    k = k + 1
                    r = unique_values[k]
                    f2 = (dist_rep == r)
                    nc_ls_1_clst = np.sum(f2 & targets)
                    nc_ls_2_clst = np.sum(f2 & ~targets)-1
                    print("0 0 1 43 57 (1, 100)")
                    print(r, nc_ls_1_clst, nc_ls_2_clst, nt_r_cls_1, nt_r_cls_2, f2.shape)

                    if   gamma * (nc_ls_2_clst / (nt_r_cls_2-1)) < (nc_ls_1_clst / nt_r_cls_1):
                        print('gamma')
                        next_value = 0
                        if (k-1) == 0:
                            r = unique_values[k]
                        else:
                            r = 0.5 * (unique_values[k-1] + unique_values[k])
                            nc_2 = nc_2 + 1
                        if r == 0:
                            r = 10**(-6)
                        r = 1*r
                        f2 = (dist_rep <= r)
                        nc_ls_1_clst = np.sum(f2 & targets)
                        nc_ls_2_clst = np.sum(f2 & ~targets) - 1               
                        q = np.where(f2 == 1)[0]
                        dr = 0
                        far = 0

                        for u in range(q.shape[0]):
                            quasi_test_p = rep_points_p[:, q[u]]

                            print('2.1520 -2.1289')
                            print(quasi_test_p)
                            print(rep_points_p.shape)
                            print(np.tile(quasi_test_p, (1, N)).shape)

                            dist_quasi_test = np.abs(np.sqrt(np.sum((rep_points_p - np.tile(quasi_test_p, (1, N))) ** 2, 1)))
                            qausi_test_class = targets[1, q[u]]
                            min1 = np.sort(dist_quasi_test, axis=2)
                            min_uniq = np.unique(min1)
                            m = 0
                            no_nereser = 0
                            while no_nereser < knn+1:
                                m = m+1
                                a1 = min_uniq(m)
                                NN = dist_quasi_test <= a1
                                no_nereser = sum(NN)
                            no_nn_c1 = sum(NN and targets)
                            no_nn_c2 = sum(NN and not targets)
                            if qausi_test_class == 1 and (no_nn_c1-1) < no_nn_c2:
                                far = far + 1
                            if qausi_test_class == 0 and no_nn_c1 < (no_nn_c2 - 1):
                                dr = dr + 1
    return [r, dr, FAR]
