import numpy as np
import math

def radiousRepX(N,No_RepPoints,patterns,targets,T,gamma,y,knn):

    NTrCls1=sum(targets)
    NTrCls2=N-NTrCls1
    M=shape(patterns,1)
    NC1=0
    NC2=0
    for i in range(No_RepPoints):
        A=where(T[:,i]==0)
        if len(A)<M:
            RepPoints_P=patterns
            RepPoints_P[A,:]=[]
            Dist_Rep=abs(math.sqrt(sum((RepPoints_P-tile(RepPoints_P[:,y],1,N))**2,1)))
            [EE_Rep]=np.msort(Dist_Rep)
            if targets(1,y)==1:
                UNQ=np.unique(EE_Rep)
                k=0
                Next=1
                while Next==1:
                    k=k+1
                    r=UNQ[k]
                    F1=(Dist_Rep==r)
                    NCls1r=sum(F1 and targets)
                    NCls2r=sum(F1 and not(targets))
                    F2=(Dist_Rep<=r)
                    NCls1clst=sum(F2 and targets)-1
                    NCls2clst=sum(F2 and not(targets))
                    if   gamma*(NCls1clst/(NTrCls1-1))<(NCls2clst/NTrCls2):
                        Next=0
                        if (k-1)==0:
                            r=UNQ[k]
                        else:
                            r=0.5*(UNQ[k-1]+UNQ[k])
                            NC1=NC1+1
                        
                        if r==0:
                            r=10**-6
                        
                        r=1*r
                        F2=(Dist_Rep<=r)
                        NCls1clst=sum(F2 and targets)-1
                        NCls2clst=sum(F2 and not(targets))
                        Q= np.where(F2==1)[1]
                        DR=0
                        FAR=0
                        for u in range(shape(Q,2)):
                            quasiTest_P=RepPoints_P[:,Q(u)]
                            Dist_quasiTest=abs(math.sqrt(sum((RepPoints_P-tile(quasiTest_P,1,N))**2,1)));
                            min1=np.msort(Dist_quasiTest,2)
                            min_Uniq=np.unique(min1)
                            m=0
                            No_nereser=0
                            while No_nereser<knn+1:
                                m=m+1
                                a1=min_Uniq[m]
                                NN=Dist_quasiTest<=a1
                                No_nereser=sum(NN)
                            
                            No_NN_C1=sum(NN and targets)
                            No_NN_C2=sum(NN and not(targets))
                            if targets[1,Q[u]]==1 and (No_NN_C1-1)>No_NN_C2:
                                DR=DR+1
                            
                            if targets[1,Q[u]]==0 and No_NN_C1>(No_NN_C2-1):
                                FAR=FAR+1
                       
            if targets(1,y)==0:
                UNQ=unique(EE_Rep)
                k=0
                Next=1
                while Next==1:
                    k=k+1
                    r=UNQ[k]
                    F1=(Dist_Rep==r)
                    NCls1r=sum(F1 and targets)
                    NCls2r=sum(F1 and not(targets))
                    F2=(Dist_Rep<=r)
                    NCls1clst=sum(F2 and targets)
                    NCls2clst=sum(F2 and not(targets))-1
                    if   gamma*(NCls2clst/(NTrCls2-1))<(NCls1clst/NTrCls1):
                        Next=0
                        if (k-1)==0:
                            r=UNQ[k]
                        else:
                            r=0.5*(UNQ[k-1]+UNQ[k])
                            NC2=NC2+1                    
                        
                        if r==0:
                            r=10**(-6)
                        
                        r=1*r
                        F2=(Dist_Rep<=r)
                        NCls1clst=sum(F2 and targets)
                        NCls2clst=sum(F2 and not(targets))-1                   
                        Q=where(F2==1)[1]
                        DR=0
                        FAR=0                    
                        for u in range(shape(Q,2)):
                            quasiTest_P=RepPoints_P[:,Q[u]]
                            Dist_quasiTest=abs(math.sqrt(sum((RepPoints_P-repmat(quasiTest_P,1,N))**2,1)))
                            quasiTest_Class=targets[1,Q[u]]
                            min1=np.msort(Dist_quasiTest,2)
                            min_Uniq=np.unique(min1)
                            m=0
                            No_nereser=0
                            while No_nereser<knn+1:
                                m=m+1
                                a1=min_Uniq(m)
                                NN=Dist_quasiTest<=a1
                                No_nereser=sum(NN)
                            
                            No_NN_C1=sum(NN and targets)
                            No_NN_C2=sum(NN and not(targets))
                            if quasiTest_Class==1 and (No_NN_C1-1)<No_NN_C2:
                                FAR=FAR+1
                            
                            if quasiTest_Class==0 and No_NN_C1<(No_NN_C2-1):
                                DR=DR+1

    return [r, DR, FAR]
