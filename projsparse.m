function A=projsparse(row,col,S,U)
%% projsparse:
% Projects the n-by-n rank-r matrix Y=U*S*U' onto the sparsity pattern 
% determined by row and col. The output A is a sparse matrix.
% It assumes that U is real n-by-r and S is real r-by-r symmetric and  
% invertible, with r<=n.

    %% IMPLEMENTATION WITHOUT FOR CYCLES
    n=size(U,1);
    US=U*S;
    a=sum(((U(row,:).*US(col,:))),2);
    A=sparse(row,col,a,n,n);
        
end