import numpy as np
import math

def ClassSimM(test, N, patterns, targets, fstar, gamma, knn):

    n_nt_cls_l = np.sum(targets) # count the number of targets
    n_nt_cls_2 = N-n_nt_cls_l # fidn the remaining somethings
    M = patterns.shape[0] 
    NC1 = 0
    NC2 = 0
    S = np.Inf * np.ones((1, N))

    NoNNC1knn = np.zeros((1,N))
    NoNNC2knn = np.zeros((1,N))
    NoNNC1 = np.zeros((1,N))
    NoNNC2 = np.zeros((1,N))
    radious = np.zeros((1,N))
    for i in range(N) :
        
        XpatternsPr = patterns*fstar[:, i][...,None]
        testPr = test*fstar[:, i]
        Dist = np.abs(np.sqrt(np.sum((-testPr[...,None] + XpatternsPr)**2,0)))
        # Dist = np.abs(np.sqrt(np.sum((XpatternsPr-testPr[..., None])**2,1)))
        min1 = np.msort(Dist)
        
        min_Uniq = np.unique(min1)
        m = -1
        No_nereser = 0
        while No_nereser<knn:
            m = m+1
            a1 = min_Uniq[m]
            NN = Dist <= a1
            No_nereser = np.sum(NN)
        NoNNC1knn[0, i] = np.sum(NN & targets)
        NoNNC2knn[0, i] = np.sum(NN & ~targets)
        
        A = np.where(fstar[:,i] == 0)
        if A[0].shape[0] <M:
            a_mask = np.ones(patterns.shape[0], dtype=bool)
            a_mask[A] = False
            patterns_P = patterns[a_mask]
            test_P = test[a_mask]
            # test_P[A,:] = []
            testA = patterns_P[:,i] - test_P
            Dist_test = np.abs(
                np.sqrt(
                    np.sum(
                        (patterns_P[:,i]-test_P)**2
                    ,0)
                )
            )
            Dist_pat = np.abs(
                np.sqrt(
                    np.sum(
                        (patterns_P-patterns_P[:, i][..., None])**2
                    ,0)
                )
            )
            EE_Rep = np.msort(Dist_pat)
            remove = 0
            if targets[0, i] == 1:
                UNQ = np.unique(EE_Rep)
                k = -1
                NC1 = NC1+1
                if remove !=  1:
                    Next = 1
                    while Next == 1:
                        k = k+1
                        r = UNQ[k]
                        F1 = (Dist_pat == r)
                        NoCls1r = np.sum(F1 & targets)
                        NoCls2r = np.sum(F1 & ~targets)
                        F2 = (Dist_pat <= r)
                        NoCls1clst = np.sum(F2 & targets)-1
                        NoCls2clst = np.sum(F2 & ~targets)
                        if gamma*(NoCls1clst/(n_nt_cls_l-1))<(NoCls2clst/n_nt_cls_2):
                            Next = 0
                            if (k-1) == 0:
                                r = UNQ[k]
                            else:
                                r = 0.5*(UNQ[k-1]+UNQ[k])
                            
                            if r == 0:
                                r = 10**-6
                            
                            r = 1*r
                            F2 = (Dist_pat <= r)
                            NoCls1clst = np.sum(F2 & targets)-1
                            NoCls2clst = np.sum(F2 & ~targets)
                    if Dist_test <= r:
                        patterns_P = patterns*fstar[:,i][...,None]
                        test_P = test*fstar[:,i]
                        Dist = np.abs(
                            np.sqrt(
                                np.sum(
                                    (patterns_P-test_P[...,None])**2,0
                                    )))
                        min1 = np.msort(Dist)
                        min_Uniq = np.unique(min1)
                        m = -1
                        No_nereser = 0
                        while No_nereser<knn:
                            m = m+1
                            a1 = min_Uniq[m]
                            NN = Dist <= a1
                            No_nereser = np.sum(NN)
                        
                        NoNNC1[0,i] = np.sum(NN & targets)
                        NoNNC2[0,i] = np.sum(NN & ~targets)
                        if NoNNC1[0,i]>NoNNC2[0,i]:
                            S[0,i] = 1
                            
            if targets[0,i] == 0:
                UNQ = np.unique(EE_Rep)
                k = -1
                NC2 = NC2+1
                if remove !=  1:
                    Next = 1
                    while Next == 1:
                        k = k+1
                        r = UNQ[k]
                        F1 = (Dist_pat == r)
                        NoCls1r = np.sum(F1 & targets)
                        NoCls2r = np.sum(F1 & ~targets)
                        F2 = (Dist_pat <= r)
                        NoCls1clst = np.sum(F2 & targets)
                        NoCls2clst = np.sum(F2 & ~targets)-1
                        if  gamma*(NoCls2clst/(n_nt_cls_2-1))<(NoCls1clst/n_nt_cls_l):
                            Next = 0
                            if (k-1) == 0:
                                r = UNQ[k]
                            else:
                                r = 0.5*(UNQ[k-1]+UNQ[k])
                            
                            if r == 0:
                                r = 10**-6
                            
                            r = 1*r
                            F2 = (Dist_pat <= r)
                            NoCls1clst = np.sum(F2 & targets)
                            NoCls2clst = np.sum(F2 & ~targets)-1
                       
                    if Dist_test <= r:
                        patterns_P = patterns*fstar[:,i][..., None]
                        test_P = test*fstar[:, i]
                        Dist = np.abs(
                            np.sqrt(
                                np.sum(
                                    (patterns_P-test_P[..., None])**2,0)))
                        min1 = np.msort(Dist)
                        min_Uniq = np.unique(min1)
                        m = -1
                        No_nereser = 0
                        while No_nereser<knn:
                            m = m+1
                            a1 = min_Uniq[m]
                            NN = Dist <= a1
                            No_nereser = np.sum(NN)
                    
                        NoNNC1[0,i] = np.sum(NN & targets)
                        NoNNC2[0,i] = np.sum(NN & ~targets)
                        if NoNNC2[0,i] > NoNNC1[0,i]:
                            S[0,i] = 1
        radious[0, i] = r

    Q1 = (NoNNC1) > (NoNNC2knn)
    Q2 = (NoNNC2) > (NoNNC1knn)
    S_Class1 = np.sum(Q1 & targets) / NC1
    S_Class2 = np.sum(Q2 & ~targets) / NC2

    if S_Class1 == 0 and S_Class2 == 0:
        Q1 = (NoNNC1knn) > (NoNNC2knn)
        Q2 = (NoNNC2knn) > (NoNNC1knn)
        S_Class1 = np.sum(Q1 & targets)/NC1
        test = Q2 & np.logical_not(targets)
        S_Class2 = np.sum(Q2 & np.logical_not(targets))/NC2

    return [S_Class1, S_Class2, radious]