function [lambda,mu,x,y]=eigtripletks(M,k,sigma)
%% eigtripletks:
% Computes the (k+1)-th and k-th eigenvalue with largest absolute value of 
% a symmetric positive definite matrix M. 
% It also returns the unit norm eigenvectors x and y associated to lambda 
% ((k+1)-th) and mu (k-th) respectively.
% The parameter sigma is a number close (but not equal) to 0.

    %% MAIN COMPUTATION
    h=k+1;
    [V,D]=eigs(M,h,sigma);
    [d,ind]=sort(diag(D),'ascend');
    V=V(:,ind);
    
    %% TARGET EIGENVALUES AND EIGENVECTORS FOUND
    lambda=d(h);
    mu=d(k);
    x=V(:,h);
    y=V(:,k);
    
    %% NORMALIZATION OF THE EIGENVECTORS
    x=x/norm(x);
    y=y/norm(y);
    
end