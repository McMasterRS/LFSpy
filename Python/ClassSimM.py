import numpy as np
import math

def ClassSimM(test,N,patterns,targets,fstar,gamma,knn):

    NTrClsl=sum(targets)
    NTrCls2=N-NTrClsl
    M=shape(patterns,1)
    NC1=0
    NC2=0
    S=inf*np.ones(1,N)

    NoNNC1knn=np.zeros(1,N)
    NoNNC2knn=np.zeros(1,N)
    NoNNC1=np.zeros(1,N)
    NoNNC2=np.zeros(1,N)
    radious=np.zeros(1,N)
    for i in range(N) :
        
        XpatternsPr=patterns*tile(fstar[:,i],1,N)
        testPr=test*fstar[:,i]
        Dist=abs(math.sqrt(sum((XpatternsPr-tile(testPr,1,N))**2,1)))
        [min1 , _]=np.msort(Dist,2)
        
        min_Uniq=np.unique(min1)
        m=0
        No_nereser=0
        while No_nereser<knn:
            m=m+1
            a1=min_Uniq(m)
            NN=Dist<=a1
            No_nereser=sum(NN)
        NoNNC1knn[1,i]=sum(NN and targets)
        NoNNC2knn[1,i]=sum(NN and not(targets))
        
        A=where(fstar[:,i]==0)
        if len(A)<M:
            patterns_P=patterns
            patterns_P[A,:]=[]
            test_P=test
            test_P[A,:]=[]
            Dist_test=abs(math.sqrt(sum((patterns_P[:,i]-test_P)**2,1)))
            Dist_pat=abs(math.sqrt(sum((patterns_P-tile(patterns_P[:,i],1,N))**2,1)))
            [EE_Rep , _]=np.msort(Dist_pat)
            remove=0
            if targets[1,i]==1:
                UNQ=np.unique(EE_Rep)
                k=0
                NC1=NC1+1
                if remove != 1:
                    Next=1
                    while Next==1:
                        k=k+1
                        r=UNQ[k]
                        F1=(Dist_pat==r)
                        NoCls1r=sum(F1 and targets)
                        NoCls2r=sum(F1 and not(targets))
                        F2=(Dist_pat<=r)
                        NoCls1clst=sum(F2 and targets)-1
                        NoCls2clst=sum(F2 and not(targets))
                        if gamma*(NoCls1clst/(NTrClsl-1))<(NoCls2clst/NTrCls2):
                            Next=0
                            if (k-1)==0:
                                r=UNQ[k]
                            else:
                                r=0.5*(UNQ[k-1]+UNQ[k])
                            
                            if r==0:
                                r=10**-6
                            
                            r=1*r
                            F2=(Dist_pat<=r)
                            NoCls1clst=sum(F2 and targets)-1
                            NoCls2clst=sum(F2 and not(targets))
                        
                    if Dist_test<=r:
                        patterns_P=patterns*tile(fstar[:,i],1,N)
                        test_P=test*fstar[:,i]
                        Dist=abs(math.sqrt(sum((patterns_P-repmat(test_P,1,N))**2,1)))
                        [min1 , _]=np.msort(Dist,2)
                        min_Uniq=np.unique(min1)
                        m=0
                        No_nereser=0
                        while No_nereser<knn:
                            m=m+1
                            a1=min_Uniq(m)
                            NN=Dist<=a1
                            No_nereser=sum(NN)
                        
                        NoNNC1[1,i]=sum(NN and targets)
                        NoNNC2[1,i]=sum(NN and not(targets))
                        if NoNNC1[1,i]>NoNNC2[1,i]:
                            S[1,i]=1
                            
            if targets[1,i]==0:
                UNQ=unique(EE_Rep)
                k=0
                NC2=NC2+1
                if remove !=1:
                    Next=1
                    while Next==1:
                        k=k+1
                        r=UNQ[k]
                        F1=(Dist_pat==r)
                        NoCls1r=sum(F1 and targets)
                        NoCls2r=sum(F1 and not(targets))
                        F2=(Dist_pat<=r)
                        NoCls1clst=sum(F2 and targets)
                        NoCls2clst=sum(F2 and not(targets))-1
                        if  gamma*(NoCls2clst/(NTrCls2-1))<(NoCls1clst/NTrClsl):
                            Next=0
                            if (k-1)==0:
                                r=UNQ[k]
                            else:
                                r=0.5*(UNQ[k-1]+UNQ[k])
                            
                            if r==0:
                                r=10**-6
                            
                            r=1*r
                            F2=(Dist_pat<=r)
                            NoCls1clst=sum(F2 and targets)
                            NoCls2clst=sum(F2 and not(targets))-1
                       
                    if Dist_test<=r:
                        patterns_P=patterns*tile(fstar[:,i],1,N)
                        test_P=test*fstar[:,i]
                        Dist=abs(math.sqrt(sum((patterns_P-tile(test_P,1,N))**2,1)))
                        [min1 , _]=np.msort(Dist,2)
                        min_Uniq=unique(min1)
                        m=0
                        No_nereser=0
                        while No_nereser<knn:
                            m=m+1
                            a1=min_Uniq(m)
                            NN=Dist<=a1
                            No_nereser=sum(NN)
                    
                        NoNNC1[1,i]=sum(NN and targets)
                        NoNNC2[1,i]=sum(NN and not(targets))
                        if NoNNC2[1,i]>NoNNC1[1,i]:
                            S[1,i]=1
                      
        radious[1,i]=r
    Q1=(NoNNC1)>(NoNNC2knn)
    Q2=(NoNNC2)>(NoNNC1knn)
    S_Class1=sum(Q1 and targets)/NC1
    S_Class2=sum(Q2 and not(targets))/NC2

    if S_Class1==0 and S_Class2==0:
        Q1=(NoNNC1knn)>(NoNNC2knn)
        Q2=(NoNNC2knn)>(NoNNC1knn)
        S_Class1=sum(Q1 and targets)/NC1
        S_Class2=sum(Q2 and not(targets))/NC2

    return [S_Class1, S_Class2, radious]
