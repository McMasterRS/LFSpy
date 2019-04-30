function [S_Class1, S_Class2, radious]=GH_ClassSimM(test,N,...
    patterns,targets,fstar,gamma,knn)

NTrClsl=sum(targets);
NTrCls2=N-NTrClsl;
M=size(patterns,1);
%%
NC1=0;
NC2=0;
S=Inf*ones(1,N);

NoNNC1knn=zeros(1,N);
NoNNC2knn=zeros(1,N);
NoNNC1=zeros(1,N);
NoNNC2=zeros(1,N);
radious=zeros(1,N);
parfor i=1:N
    %%
    XpatternsPr=patterns.*repmat(fstar(:,i),1,N);
    testPr=test.*fstar(:,i);
    Dist=abs(sqrt(sum((XpatternsPr-repmat(testPr,1,N)).^2,1)));
    [min , ~]=sort(Dist,2);
    
    min_Uniq=unique(min);
    m=0;
    No_nereser=0;
    while No_nereser<knn
        m=m+1;
        a1=min_Uniq(m);
        NN=Dist<=a1;
        No_nereser=sum(NN);
    end
    NoNNC1knn(1,i)=sum(and(NN,targets));
    NoNNC2knn(1,i)=sum(and(NN,not(targets)));
    %%
    A=find(fstar(:,i)==0);
    if length(A)<M
        patterns_P=patterns;
        patterns_P(A,:)=[];
        test_P=test;
        test_P(A,:)=[];
        Dist_test=abs(sqrt(sum((patterns_P(:,i)-test_P).^2,1)));
        Dist_pat=abs(sqrt(sum((patterns_P-repmat(patterns_P(:,i),1,N)).^2,1)));
        [EE_Rep , ~]=sort(Dist_pat);
        remove=0;
        if targets(1,i)==1
            UNQ=unique(EE_Rep);
            k=0;
            NC1=NC1+1;
            if remove~=1
                Next=1;
                while Next==1
                    k=k+1;
                    r=UNQ(k);
                    F1=(Dist_pat==r);
                    NoCls1r=sum(and(F1,targets));
                    NoCls2r=sum(and(F1,not(targets)));
                    F2=(Dist_pat<=r);
                    NoCls1clst=sum(and(F2,targets))-1;
                    NoCls2clst=sum(and(F2,not(targets)));
                    if    gamma*(NoCls1clst/(NTrClsl-1))<(NoCls2clst/NTrCls2)
                        Next=0;
                        if (k-1)==0
                            r=UNQ(k);
                        else
                            r=0.5*(UNQ(k-1)+UNQ(k));
                        end
                        if r==0
                            r=1e-6;
                        end
                        r=1*r;
                        F2=(Dist_pat<=r);
                        NoCls1clst=sum(and(F2,targets))-1;
                        NoCls2clst=sum(and(F2,not(targets)));
                    end
                end
                if Dist_test<=r
                    patterns_P=patterns.*repmat(fstar(:,i),1,N);
                    test_P=test.*fstar(:,i);
                    Dist=abs(sqrt(sum((patterns_P-repmat(test_P,1,N)).^2,1)));
                    [min , ~]=sort(Dist,2);
                    min_Uniq=unique(min);
                    m=0;
                    No_nereser=0;
                    while No_nereser<knn
                        m=m+1;
                        a1=min_Uniq(m);
                        NN=Dist<=a1;
                        No_nereser=sum(NN);
                    end
                    NoNNC1(1,i)=sum(and(NN,targets));
                    NoNNC2(1,i)=sum(and(NN,not(targets)));
                    if NoNNC1(1,i)>NoNNC2(1,i)
                        S(1,i)=1;
                    end
                end
            end
        end
        if targets(1,i)==0
            UNQ=unique(EE_Rep);
            k=0;
            NC2=NC2+1;
            if remove~=1
                Next=1;
                while Next==1
                    k=k+1;
                    r=UNQ(k);
                    F1=(Dist_pat==r);
                    NoCls1r=sum(and(F1,targets));
                    NoCls2r=sum(and(F1,not(targets)));
                    F2=(Dist_pat<=r);
                    NoCls1clst=sum(and(F2,targets));
                    NoCls2clst=sum(and(F2,not(targets)))-1;
                    if  gamma*(NoCls2clst/(NTrCls2-1))<(NoCls1clst/NTrClsl)
                        Next=0;
                        if (k-1)==0
                            r=UNQ(k);
                        else
                            r=0.5*(UNQ(k-1)+UNQ(k));
                        end
                        if r==0
                            r=1e-6;
                        end
                        r=1*r;
                        F2=(Dist_pat<=r);
                        NoCls1clst=sum(and(F2,targets));
                        NoCls2clst=sum(and(F2,not(targets)))-1;
                    end
                end
                if Dist_test<=r
                    patterns_P=patterns.*repmat(fstar(:,i),1,N);
                    test_P=test.*fstar(:,i);
                    Dist=abs(sqrt(sum((patterns_P-repmat(test_P,1,N)).^2,1)));
                    [min , ~]=sort(Dist,2);
                    min_Uniq=unique(min);
                    m=0;
                    No_nereser=0;
                    while No_nereser<knn
                        m=m+1;
                        a1=min_Uniq(m);
                        NN=Dist<=a1;
                        No_nereser=sum(NN);
                    end
                    NoNNC1(1,i)=sum(and(NN,targets));
                    NoNNC2(1,i)=sum(and(NN,not(targets)));
                    if NoNNC2(1,i)>NoNNC1(1,i)
                        S(1,i)=1;
                    end
                end
            end
        end
    end
    radious(1,i)=r;
end
Q1=(NoNNC1)>(NoNNC2knn);
Q2=(NoNNC2)>(NoNNC1knn);
S_Class1=sum(and(Q1,targets))/NC1;
S_Class2=sum(and(Q2,not(targets)))/NC2;

if S_Class1==0 && S_Class2==0
    Q1=(NoNNC1knn)>(NoNNC2knn);
    Q2=(NoNNC2knn)>(NoNNC1knn);
    S_Class1=sum(and(Q1,targets))/NC1;
    S_Class2=sum(and(Q2,not(targets)))/NC2;
end





