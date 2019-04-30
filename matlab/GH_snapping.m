function [Bratio,TT_binary,feasibo,R]=GH_snapping...
    (NRRP,M,i,BB,b1,gamma,TT,patterns,targets,N,a,b,knn)

No_C1=sum(targets);
No_C2=sum(not(targets));
TT_binary_temp=zeros(M,1);
rng(20);
r=rand(M,NRRP);
A=r<=repmat(TT,1,NRRP);
A_unique=logical((unique(A','rows'))');
No_unq=size(A_unique,2);
radious=zeros(1,No_unq);
feasib=zeros(1,No_unq);
wit_dist=Inf*ones(1,No_unq);
Btw_dist=-Inf*ones(1,No_unq);
DR=zeros(1,No_unq);
FAR=zeros(1,No_unq);
for y=1:No_unq
    TT_binary_temp(A_unique(:,y))=1;
    TT_binary_temp(not(A_unique(:,y)))=0;
    if sum(BB(1:end,:)*TT_binary_temp>b1(1:end))==0 % if atleast one feature is active and no more than maxNoFea
        feasib(1,y)=1;
        wit_dist(y)=a*TT_binary_temp;
        Btw_dist(y)=b*TT_binary_temp;
        [radious(y), DR(y), FAR(y)]=GH_radiousRepX(N,1,patterns,targets,TT_binary_temp,gamma,i,knn);
    end
end
evaluation_Cri_C1=DR/No_C1-FAR/No_C2;
evaluation_Cri_C2=DR/No_C2-FAR/No_C1;
[~,b1]=min(wit_dist);
TT_binary=A_unique(:,b1);
feasibo=feasib(1,b1);
R=radious(1,b1);
if targets(1,i)==1
    Bratio=evaluation_Cri_C1(b1);
end
if targets(1,i)==0
    Bratio=evaluation_Cri_C2(b1);
end


