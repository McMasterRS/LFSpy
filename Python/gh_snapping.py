import numpy as np
from radious_rep_x import radious_rep_x

# TILE PROBLEM FIX https://stackoverflow.com/questions/32238227/numpy-tile-did-not-work-as-matlab-repmat

def gh_snapping(NRRP, M, i, BB, b1, gamma, TT, patterns, targets, N, a, b, knn):
    print('Hello Snapping')

    No_C1 = np.sum(targets)
    No_C2 = np.sum(np.logical_not(targets))
    TT_binary_temp = np.zeros((M, 1))
    np.random.seed(20)
    r = np.random.rand(M, NRRP)
    A = r <= np.tile(TT, (1, NRRP))
    A_unique = np.unique(A.T, axis=0).T # finds all of the unique rows in the array


    No_unq = A_unique.shape[1]
    radious = np.zeros((1, No_unq))
    feasib = np.zeros((1, No_unq))
    wit_dist = np.inf * np.ones((1, No_unq)) # within distance
    Btw_dist = -1 * np.inf * np.ones((1, No_unq)) # between distance
    DR = np.zeros((1, No_unq))
    FAR = np.zeros((1, No_unq))

    # No_C1 == 43, No_C2 == 57, r == [160x2000], A == [160x2000], A_unique == 160x4
    print("% 43 57 (160, 2000) (160, 2000) (160, 4) 4")
    print("#", No_C1, No_C2, r.shape, A.shape, A_unique.shape, No_unq)

    for y in range(No_unq):
        # TT_binary_temp == [160x1], 
        TT_binary_temp[A_unique[:, y]] = 1 # fill all slots where there are unique rows with 1
        # TT_binary_temp[np.logical_not(A_unique[:, y])] = 0 # fill the remaining with 0

        print("% (160, 1) 0:0 -> 1:1 -> 2:2 -> 3:2")
        print("#", np.sum(TT_binary_temp), TT_binary_temp.shape, y, ':', np.sum(TT_binary_temp))

        # @ symbol is matrix multiplication and is equivilant to just * (not .*) in matlab
        if np.sum(BB @ TT_binary_temp > b1[..., None]) == 0:  # if atleast one feature is active and no more than maxNoFea
            feasib[0, y] = 1
            wit_dist[0, y] = a @ TT_binary_temp
            Btw_dist[0, y] = b @ TT_binary_temp
            [radious[y], DR[y], FAR[y]] = radious_rep_x(N, 1, patterns, targets, TT_binary_temp, gamma, i, knn)
        
    evaluation_Cri_C1 = DR/No_C1-FAR/No_C2
    evaluation_Cri_C2 = DR/No_C2-FAR/No_C1
    b1 = np.min(wit_dist)
    TT_binary = A_unique[:, b1]
    feasibo = feasib[1, b1]
    R = radious[1,b1]

    if targets[1,i] == 1:
        Bratio = evaluation_Cri_C1[b1]

    if targets[1,i] == 0:
        Bratio = evaluation_Cri_C2[b1]

    return Bratio, TT_Binary, feasibo, R