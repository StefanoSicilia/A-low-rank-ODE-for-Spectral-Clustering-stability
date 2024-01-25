function L=LapSparse(W)
%% LapSparse:
% Computes the sparse Laplacian L of the sparse weight matrix W.
    
    one=sparse(ones(size(W,1),1));
    L=diag(W*one)-W;

end