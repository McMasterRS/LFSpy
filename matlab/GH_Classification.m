function [S_Class1_sphereKNN,S_Class2_sphereKNN]=GH_Classification...
    (patterns,targets,N,test,fstar,gamma,knn)
H=size(test,2);
S_Class1_sphereKNN=zeros(1,H);
S_Class2_sphereKNN=zeros(1,H);
parfor t=1:H
    [S_Class1_sphereKNN(1,t), S_Class2_sphereKNN(1,t),~]=...
        GH_ClassSimM(test(:,t),N,patterns,targets,fstar,gamma,knn);
end
end


