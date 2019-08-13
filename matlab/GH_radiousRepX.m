function [r, DR, FAR]=GH_radiousRepX(N,No_RepPoints,patterns,targets,T,gamma,y,knn)

NTrCls1=sum(targets);
NTrCls2=N-NTrCls1;
M=size(patterns,1);
NC1=0;
NC2=0;
for i=1:No_RepPoints
    A=find(T(:,i)==0);
    if length(A)<M
        RepPoints_P=patterns;
        RepPoints_P(A,:)=[];
        Dist_Rep=abs(sqrt(sum((RepPoints_P-repmat(RepPoints_P(:,y),1,N)).^2,1)));
        [EE_Rep , ~]=sort(Dist_Rep);
        if targets(1,y)==1
            UNQ=unique(EE_Rep);
            k=0;
            Next=1;
            while Next==1
                k=k+1;
                r=UNQ(k);
                F1=(Dist_Rep==r);
                NCls1r=sum(and(F1,targets));
                NCls2r=sum(and(F1,not(targets)));
                F2=(Dist_Rep<=r);
                NCls1clst=sum(and(F2,targets))-1;
                NCls2clst=sum(and(F2,not(targets)));
                if   gamma*(NCls1clst/(NTrCls1-1))<(NCls2clst/NTrCls2)
                    Next=0;
                    if (k-1)==0
                        r=UNQ(k);
                    else
                        r=0.5*(UNQ(k-1)+UNQ(k));
                        NC1=NC1+1;
                    end
                    if r==0
                        r=1e-6;
                    end
                    r=1*r;
                    F2=(Dist_Rep<=r);
                    NCls1clst=sum(and(F2,targets))-1;
                    NCls2clst=sum(and(F2,not(targets)));
                    [~, Q]=find(F2==1);
                    DR=0;
                    FAR=0;
                    for u=1:size(Q,2)
                        quasiTest_P=RepPoints_P(:,Q(u));
                        Dist_quasiTest=abs(sqrt(sum((RepPoints_P-repmat(quasiTest_P,1,N)).^2,1)));
                        [min , ~]=sort(Dist_quasiTest,2);
                        min_Uniq=unique(min);
                        m=0;
                        No_nereser=0;
                        while No_nereser<knn+1
                            m=m+1;
                            a1=min_Uniq(m);
                            NN=Dist_quasiTest<=a1;
                            No_nereser=sum(NN);
                        end
                        No_NN_C1=sum(and(NN,targets));
                        No_NN_C2=sum(and(NN,not(targets)));
                        if targets(1,Q(u))==1 && (No_NN_C1-1)>No_NN_C2
                            DR=DR+1;
                        end
                        if targets(1,Q(u))==0 && No_NN_C1>(No_NN_C2-1)
                            FAR=FAR+1;
                        end
                    end
                end
            end
        end
        if targets(1,y)==0
            UNQ=unique(EE_Rep);
            k=0;
            Next=1;
            while Next==1
                k=k+1;
                r=UNQ(k);
                F1=(Dist_Rep==r);
                NCls1r=sum(and(F1,targets));
                NCls2r=sum(and(F1,not(targets)));
                F2=(Dist_Rep<=r);
                NCls1clst=sum(and(F2,targets));
                NCls2clst=sum(and(F2,not(targets)))-1;
                if   gamma*(NCls2clst/(NTrCls2-1))<(NCls1clst/NTrCls1)
                    Next=0;
                    if (k-1)==0
                        r=UNQ(k);
                    else
                        r=0.5*(UNQ(k-1)+UNQ(k));
                        NC2=NC2+1;                        
                    end
                    if r==0
                        r=1e-6;
                    end
                    r=1*r;
                    F2=(Dist_Rep<=r);
                    NCls1clst=sum(and(F2,targets));
                    NCls2clst=sum(and(F2,not(targets)))-1;                    
                    [~, Q]=find(F2==1);
                    DR=0;
                    FAR=0;                    
                    for u=1:size(Q,2)
                        quasiTest_P=RepPoints_P(:,Q(u));
                        Dist_quasiTest=abs(sqrt(sum((RepPoints_P-repmat(quasiTest_P,1,N)).^2,1)));
                        quasiTest_Class=targets(1,Q(u));
                        [min , ~]=sort(Dist_quasiTest,2);
                        min_Uniq=unique(min);
                        m=0;
                        No_nereser=0;
                        while No_nereser<knn+1
                            m=m+1;
                            a1=min_Uniq(m);
                            NN=Dist_quasiTest<=a1;
                            No_nereser=sum(NN);
                        end
                        No_NN_C1=sum(and(NN,targets));
                        No_NN_C2=sum(and(NN,not(targets)));
                        if quasiTest_Class==1 && (No_NN_C1-1)<No_NN_C2
                            FAR=FAR+1;
                        end
                        if quasiTest_Class==0 && No_NN_C1<(No_NN_C2-1)
                            DR=DR+1;
                        end
                    end
                end
            end
        end
    end
end
disp('here')