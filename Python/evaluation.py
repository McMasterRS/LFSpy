import numpy as np
from gh_lp_opt import gh_lp_opt
from gh_snapping import gh_snapping
# from radiousRepX import radiousRepX
import cProfile, pstats, io
pr = cProfile.Profile()


def evaluation (alpha, NBeta, N, EpsilonMax, b, a, patterns, targets, gamma, NRRP, knn):

    M = b.shape[1]
    TRTemp = np.zeros((M, N, NBeta))
    TBTemp = np.zeros((M, N, NBeta))
    Bratio = -2*np.ones((N, NBeta))
    feasib = np.zeros((N, NBeta))
    radiuos = np.zeros((N, NBeta))

    for i in range(N) :
        for n in range(NBeta):
            TT, BB, b1, exit_flag = gh_lp_opt(NBeta, n+1, EpsilonMax[i, 0], b[i, :], a[i, :], M, alpha)
            if exit_flag == 1:
                Bratio[i, n], T_temp, feasib[i, n], radiuos[i, n] = gh_snapping(NRRP, M, i, BB, b1, gamma, TT, patterns, targets, N, a[i, :], b[i, :], knn)
                if feasib[i, n] == 1:
                    TBTemp[:, i, n] = T_temp
                    TRTemp[:, i, n] = TT[:, 0]
        pass

    return [TBTemp, TRTemp, Bratio, feasib, radiuos]