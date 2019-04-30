function [DCls2,DCls1,erCls1,erCls2,ErClassification]=GH_accuracy...
    (S_Class1,S_Class2,TestLables)
NTest=size(TestLables,2);
ErClassification=zeros(1,1);
erCls2=zeros(1,1);
erCls1=zeros(1,1);
DCls1=zeros(1,1);
DCls2=zeros(1,1);
NCls1Tst=sum(TestLables);
NCls2Test=NTest-NCls1Tst;
n=1;

DQ0=TestLables==0;
SS1=S_Class1(1,:,n);
SS2=S_Class2(1,:,n);
DQ1=(SS1)>(SS2);
DQ1=double(DQ1);
DQ1(DQ0)=0;
DCls1(n)=sum(sum(DQ1));
erCls1(n)=1-DCls1(n)/NCls1Tst;
erCls1(n)=erCls1(n)*100;

DQ0_SCZ=TestLables==0;
SS1_SCZ=S_Class1(1,:,n);
SS2_SCZ=S_Class2(1,:,n);
DQ1_SCZ=(SS1_SCZ)<(SS2_SCZ);
DQ1_SCZ=double(DQ1_SCZ);
DQ1_SCZ(not(DQ0_SCZ))=0;
DCls2(n)=sum(sum(DQ1_SCZ));
erCls2(n)=1-DCls2(n)/NCls2Test;
erCls2(n)=erCls2(n)*100;

ErClassification(n)=1-(DCls1(n)+DCls2(n))/NTest;
ErClassification(n)=ErClassification(n)*100;





