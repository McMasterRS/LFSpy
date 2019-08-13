import numpy as np
from gh_lp_opt import gh_lp_opt
from gh_snapping import gh_snapping
# from radiousRepX import radiousRepX
import cProfile, pstats, io
pr = cProfile.Profile()
from scipy.io import loadmat


def evaluation (alpha, NBeta, N, EpsilonMax, b, a, patterns, targets, gamma, NRRP, knn):

    def validate_gh_snapping(b_ratio, t_temp, feasib, radious, i, j):
        data = loadmat('./data/gh_snapping')
        _bratio = data['Bratio'][i, j]
        _feasib = data['feasib'][i, j]
        _radious = data['radiuos'][i, j]
        _diff_bratio = np.isclose(b_ratio,_bratio)
        _diff_feasib = feasib == feasib
        _diff_radious = np.isclose(radious, _radious)
        print("validate_gh_snapping", i, j, _diff_bratio, _diff_feasib, _diff_radious)
        if _diff_bratio == False or _diff_feasib == False or _diff_radious == False:
            print('Uh Oh')
        return _diff_bratio

    def validate_gh_lp_opt(TT, BB, b1, exit_flag, i, j):
        data = loadmat('./data/TT_dump')
        a = TT
        b = data['TT_dump'][i, n, :][...,None]
        _a = np.isclose(a, b, atol=0.1, rtol=0.000001)
        print('validate_gh_lp_opt', i, j, _a.all())
        if _a.all() == False:
            _where = np.where(_a == False)
            diff = (a[_where], b[_where]) 
            print('Uh Oh', diff, exit_flag)
        return _a
    
    M = b.shape[1]
    TRTemp = np.zeros((M, N, NBeta))
    TBTemp = np.zeros((M, N, NBeta))
    Bratio = -2*np.ones((N, NBeta))
    feasib = np.zeros((N, NBeta))
    radiuos = np.zeros((N, NBeta))

    for i in range(N) :
        for n in range(NBeta):
            if i == 5 and n == 3:
                TT, BB, b1, exit_flag = gh_lp_opt(NBeta, n+1, EpsilonMax[i, 0], b[i, :], a[i, :], M, alpha)
                validate_gh_lp_opt(TT, BB, b1, exit_flag, i, n)
                # data = loadmat('./data/TT_dump')
                # TT = data['TT_dump'][i, n, :][..., None] # Fixing my TT values gives correct results for no_unq

                if exit_flag == 1:
                    Bratio[i, n], T_temp, feasib[i, n], radiuos[i, n] = gh_snapping(NRRP, M, i, BB, b1, gamma, TT, patterns, targets, N, a[i, :], b[i, :], knn, n)
                    validate_gh_snapping(Bratio[i, n], T_temp, feasib[i, n], radiuos[i, n], i, n)
                    if feasib[i, n] == 1:
                        TBTemp[:, i, n] = T_temp
                        TRTemp[:, i, n] = TT[:, 0]

    return [TBTemp, TRTemp, Bratio, feasib, radiuos]