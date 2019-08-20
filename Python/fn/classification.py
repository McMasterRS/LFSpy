import numpy as np
from fn.class_sim_m import class_sim_m

def classification(patterns, targets, N, test, fstar, params):
    """ """

    gamma = params['gamma']
    knn = params['knn']
    H = test.shape[1]
    s_class_1_sphere_knn = np.zeros((1, H))
    s_class_2_sphere_knn = np.zeros((1, H))

    for t in range(H) :
        s_class_1_sphere_knn[0, t], s_class_2_sphere_knn[0, t], _ = class_sim_m(test[:, t], N, patterns, targets, fstar, gamma, knn)

    return [s_class_1_sphere_knn, s_class_2_sphere_knn]
