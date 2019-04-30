function [fstar,fstarLin,ErCls1,ErCls2,ErClassification] = LFS(Train, TrainLables, Test, TestLables, Para)

%LFS: local feature selection and classification
%% Citation: [1] and [2]
%[1] N. Armanfard, JP. Reilly, and M. Komeili, "Local Feature Selection for Data Classification", IEEE Trans. on Pattern Analysis and Machine Intelligence, vol. 38, no. 6, pp. 1217-1227, 2016.
%[2] N. Armanfard, JP. Reilly, and M. Komeili, "Logistic Localized Modeling of the Sample Space for Feature Selection and Classification", IEEE Transactions on Neural Networks and Learning Systems, vol. 29, no. 5, pp. 1396-1413, 2018
%--------------------------------------------------------------------------
%INPUT:
%       Train (M by N):        training data : [x1,x2,...xN] Each column is an observation; M is number of candidate fetatures.
%       TrainLables (1 by N):  class label = {0,1}
%       Test (M by K):         M is number of candidate features; K is number of test points.
%       TestLables (1 by K):   class label = {0,1}
%       Para:  parameters.
%           Para.gamma:   impurity level (default: 0.2)
%           Para.tau:     number of iterations (default: 2)
%           Para.sigma:   controlls neighboring samples weighting (default: 1)
%           Para.alpha:   maximum number of selected feature for each representative point
%           Para.NBeta:   numer of distinct \beta (default: 20)
%           Para.NRRP:    number of iterations for randomized rounding process (defaul 2000)
%OUTPUT:
%       fstar (M by N):   selected features for each representative point; if fstar(i,j)=1, ith feature is selected for jth representative point.
%       fstarLin (M by N):it is fsatr before applying randomized rounding process
%       ErCls1:           it is the classification error (in percent) associated to the input test points with class label 1.
%       ErCls2:           it is the classification error (in percent) associated to the input test points with class label 0.
%       ErClassification: it is the total error (in percent) for the entire input test points.
%--------------------------------------------------------------------------
%% by Narges Armanfard
    %update history: April 23 2019
%% ==========================================================================
%% parameters
gamma=Para.gamma;
tau=Para.tau;
alpha=Para.alpha;
sigma=Para.sigma;
NBeta=Para.NBeta;
NRRP=Para.NRRP;
knn=1;

N=size(Train,2);
M=size(Train,1);
fstar=zeros(M,N);
fstarLin=zeros(M,N);
for j=1:tau
    [b, a, EpsilonMax]=GH_FES(alpha,M,N,Train,TrainLables,fstar,sigma);
    
    [TBTemp,TRTemp,Bratio,feasib,~]=GH_evaluation(alpha,NBeta,N,EpsilonMax,b,a,Train,TrainLables,gamma,NRRP,knn);
    
    W1=feasib==0; Bratio(W1)=-Inf;
    [~, I1]=max(Bratio,[],2);
    parfor i=1:N
        fstar(:,i)=TBTemp(:,i,I1(i));
        fstarLin(:,i)=TRTemp(:,i,I1(i));
    end
end

[SClass1,SClass2]=GH_Classification(Train,TrainLables,N,Test,fstar,gamma,knn);

[~,~, ErCls1,ErCls2,ErClassification]=GH_accuracy(SClass1,SClass2,TestLables);
end