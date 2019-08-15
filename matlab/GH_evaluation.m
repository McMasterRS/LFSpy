function [TBTemp,TRTemp,Bratio,feasib,radiuos]=GH_evaluation...
    (alpha,NBeta,N,EpsilonMax,b,a,patterns,targets,gamma,NRRP,knn)

M=size(b,2);
%%
TRTemp=zeros(M,N,NBeta);
TBTemp=zeros(M,N,NBeta);
Bratio=-2*ones(N,NBeta);
feasib=zeros(N,NBeta);
radiuos=zeros(N,NBeta);
TT_dump = zeros(N,NBeta,160);
BB_dump = zeros(N,NBeta,160);
%parfor i=1:N
for i=1:N
    for n=1:NBeta
        [TT, BB, b1, ~, ~, exitflag]=GH_LP_Opt(NBeta,n,EpsilonMax(i,1),b(i,:),a(i,:),...
            M,alpha);
        if exitflag==1
            TT_dump(i,n,:) = TT;
            [Bratio(i,n),T_temp,feasib(i,n),radiuos(i,n)]=GH_snapping...
                (NRRP,M,i,BB,b1,gamma,TT,patterns,targets,N,a(i,:),b(i,:),knn, n);   
            if feasib(i,n)==1
                TBTemp(:,i,n)=T_temp;
                TRTemp(:,i,n)=TT;
            end
        end                
    end
end
end









