import numpy as np
from radious_rep_x import radious_rep_x
from scipy.io import loadmat

# TILE PROBLEM FIX https://stackoverflow.com/questions/32238227/numpy-tile-did-not-work-as-matlab-repmat

def gh_snapping(NRRP, M, i, BB, b1, gamma, TT, patterns, targets, N, a, b, knn, j):

    def validate_radious_rep_x(radious, DR, FAR, j):
        pass

    No_C1 = np.sum(targets)
    No_C2 = np.sum(np.logical_not(targets))
    TT_binary_temp = np.zeros((M, 1))
    np.random.seed(20)
    # r = np.random.rand(M, NRRP)
    r = loadmat('./data/r')['r']
    A = r <= np.tile(TT, (1, NRRP))
    A_unique = np.unique(A.T, axis=0).T # finds all of the unique rows in the array
    No_unq = A_unique.shape[1] # ITS SOMETHING TO DO WITH THE NUMBER OF UNIQUE VALUES
    radious = np.zeros((1, No_unq))
    feasib = np.zeros((1, No_unq))
    wit_dist = np.inf * np.ones((1, No_unq)) # within distance
    Btw_dist = -1 * np.inf * np.ones((1, No_unq)) # between distance
    DR = np.zeros((1, No_unq))
    FAR = np.zeros((1, No_unq))

    for y in range(No_unq):
        TT_binary_temp[A_unique[:, y]] = 1 # fill all slots where there are unique rows with 1
        TT_binary_temp[np.logical_not(A_unique[:, y])] = 0 # fill all slots where there are not unique rows with 1
        if np.sum(BB @ TT_binary_temp > b1) == 0:  # if atleast one feature is active and no more than maxNoFea
            feasib[0, y] = 1
            wit_dist[0, y] = a @ TT_binary_temp
            Btw_dist[0, y] = b @ TT_binary_temp
            [radious[0, y], DR[0, y], FAR[0, y]] = radious_rep_x(N, 1, patterns, targets, TT_binary_temp, gamma, i, knn)
    evaluation_Cri_C1 = DR/No_C1-FAR/No_C2
    evaluation_Cri_C2 = DR/No_C2-FAR/No_C1
    b1 = np.argmin(wit_dist)
    TT_binary = A_unique[:, b1]
    feasibo = feasib[0, b1]
    R = radious[0, b1]

    if targets[0, i] == 1:
        Bratio = evaluation_Cri_C1[0, b1]

    if targets[0,i] == 0:
        Bratio = evaluation_Cri_C2[0, b1]

    return Bratio, TT_binary, feasibo, R