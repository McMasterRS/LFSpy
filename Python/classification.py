import numpy as np
from ClassSimM import ClassSimM

def classification(patterns, targets, N, test, fstar, gamma, knn):
    """ """
    H = test.shape[1]
    s_class_1_sphere_knn = np.zeros((1, H))
    s_class_2_sphere_knn = np.zeros((1, H))

    for t in range(H) :
        s_class_1_sphere_knn[0, t], s_class_2_sphere_knn[0, t], _ = ClassSimM(test[:, t], N, patterns, targets, fstar, gamma, knn)

    return [s_class_1_sphere_knn, s_class_2_sphere_knn]
