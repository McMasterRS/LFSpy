function [b,a,EpsilonMax]=GH_FES...
    (alpha,M,N,patterns,targets,fstar,sigma)
    
[b(:,:), a(:,:), EpsilonMax(:,1)]=...
    GH_bisec(M,N,patterns,targets,fstar,...
    alpha,sigma);
end