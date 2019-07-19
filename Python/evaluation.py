import numpy as np
import radiousRepX as radiousRepX

def evaluation (alpha,NBeta,N,EpsilonMax,b,a,patterns,targets,gamma,NRRP,knn):
    M=shape(b,2)

    TRTemp = np.zeros(M,N,NBeta)
    TBTemp = np.zeros(M,N,NBeta)
    Bratio = -2*np.ones(N,NBeta)
    feasib = np.zeros(N,NBeta)
    radiuos = np.zeros(N,NBeta)
    for i in range(N) : 
        for n in range(NBeta) :
            
            beta=1/NBeta*n
            epsilon=beta*EpsilonMax

            b1=list(range(alpha, -1*epsilon,-1))
            

            #BB=[np.ones(1,M);-1*np.ones(1,M);-b] Not sure about this part
            BB = list(range(np.ones(1,M), -b, -1*np.ones(1,M)))
            lb=np.zeros(M,1)
            ub=np.ones(M,1)
            
            
            res = scipy.optimize.linprog(a.conj().T,BB,b1,lb,ub,method='interior-point')
            if res.success==1:
               

                No_C1=sum(targets)
                No_C2=sum(not(targets))
                TT_binary_temp=np.zeros(M,1)
                
                r=np.rand(M,NRRP)
                A=r<=tile(TT,(1,NRRP))
                A_unique=logical((np.unique(A.conj().T,'rows')).conj().T); # Having trouble implementing the logical function
                                 
                No_unq=shape(A_unique,2)
                radious=np.zeros(1,No_unq)
                feasib=np.zeros(1,No_unq)
                wit_dist=inf*np.ones(1,No_unq)
                Btw_dist=-inf*np.ones(1,No_unq)
                DR=np.zeros(1,No_unq)
                FAR=np.zeros(1,No_unq)
                for y in range(No_unq):
                    TT_binary_temp[A_unique[:,y]]=1
                    TT_binary_temp[not(A_unique[:,y])]=0
                    if sum(BB[1,-1,-1]*TT_binary_temp>b1[1,-1])==0:  #if atleast one feature is active and no more than maxNoFea
                        feasib[1,y]=1
                        wit_dist[y]=a*TT_binary_temp
                        Btw_dist[y]=b*TT_binary_temp
                        [radious[y], DR[y], FAR[y]]=radiousRepX(N,1,patterns,targets,TT_binary_temp,gamma,i,knn);
                    
                evaluation_Cri_C1=DR/No_C1-FAR/No_C2
                evaluation_Cri_C2=DR/No_C2-FAR/No_C1
                b1=min[wit_dist][1]
                TT_binary=A_unique[:,b1]
                feasibo=feasib[1,b1]
                R=radious[1,b1]
                if targets[1,i]==1:
                    Bratio=evaluation_Cri_C1[b1]
                
                if targets[1,i]==0:
                    Bratio=evaluation_Cri_C2[b1]
                
                if feasib[i,n]==1:
                    TBTemp[:,i,n]=T_temp
                    TRTemp[:,i,n]=TT

    return [TBTemp,TRTemp,Bratio,feasib,radiuos]
