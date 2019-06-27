import numpy as np
from scipy.optimize import fsolve

def FES(alpha, M, N, patterns, targets, fstar, sigma):


    b = np.zeros(N,M)
    a = np.zeros(N,M)
    EpsilonMax = np.zeros(N,1)
    weight = np.zeros(N,N)

for i in range(1,N):
    cls1=targets(1,i)
    Temp_Tr=patterns
    Temp_Tr[:,i]=[]
    Temp_class=targets
    Temp_class[:,i]=[]
    x=patterns[:,i]
    Ncls1=sum(Temp_class)
    Ncls2=np.shape(Temp_class,2)-Ncls1
    A=np.zeros(M,Ncls1)
    B=np.zeros(M,Ncls2)
    k=1
    m=1
    f_cls1_temp=np.zeros(M,1)
    f_cls2_temp=np.zeros(M,1)
    f_Sparsity=np.zeros(M,1)

    for j in range(1,np.shape(Temp_Tr, 2)):
        if Temp_class(1,j) ==1:
            A[:,k] = Temp_Tr[:,j]
            k = k + 1
        if Temp_class(1, j) ==0:
            B[:,m] = Temp_Tr[:,j]
            m = m+1
    C=repmat(x,1,Ncls1)-A
    D=repmat(x,1,Ncls2)-B
    ro=C**2
    theta=D**2
    w11=np.zeros(1,Ncls1)
    w22=np.zeros(1,Ncls2)
    w=np.zeros(N,N)
    w_ave=np.zeros(1,N)
    V=np.array[1:N]
    V(i)=[]

    for p in range(1,N):
        Dist_ro=sqrt(sum(repmat(fstar[:,p],[1,Ncls1])*ro,1))
        Dist_alpha=sqrt(sum(repmat(fstar[:,p],[1,Ncls2])*theta,1))
        w11=exp(-1*(Dist_ro-min(Dist_ro))**2/sigma)
        w22=exp(-1*(Dist_alpha-min(Dist_alpha))**2/sigma)
        w(p,V)=[w11[1,:],w22[1,:]]
    
    w_ave=mean(w)
    w_ave(i)=[]  
    w1=w_ave[1,1:Ncls1]
    w2=w_ave[1,Ncls1+1:end]
    w3=0
    normalized_w1=w1/sum(w1)
    normalized_w2=w2/sum(w2)
    wandering=0
    w1=(1-wandering)*normalized_w1+wandering*np.random.rand(1,size(w1,2))
    w2=(1-wandering)*normalized_w2+wandering*np.random.rand(1,size(w2,2))   
    normalized_w1=w1/sum(w1)
    normalized_w2=w2/sum(w2)
    weight[:,i]=[normalized_w1.T,normalized_w2.T,w3]
    w=weight[:,i]

    if cls1==1:
        for n in range(1,Ncls1):
            f_cls1_temp=f_cls1_temp+w(n)*ro[:,n]
        for n in range (1,Ncls2):
            f_cls2_temp=f_cls2_temp+w(n+Ncls1)*theta[:,n]
        f_Sparsity=w(end)*np.ones(M,1)
        f_temp=f_Sparsity+f_cls2_temp
        b[i,:]=f_temp.T/Ncls2
        a[i,:]=f_cls1_temp.T/Ncls1

    BB=[np.ones(1,M),-1*np.ones(1,M)]
    b1=[alpha,-1]
    lb=np.zeros(M,1)
    ub=np.ones(M,1)
    res = scipy.optimize.linprog(-b[i,:].T,BB,b1,lb,ub,method='interior-point')
    if res.success != 1
       print 'Not feasible'
    if res.success == 1
       EpsilonMax[i,1]= -1*res.fun

    return [b, a, EpsilonMax]
