function [c,P]=costPartition(A,v,k)
%% costPartition: 
% Given a graph with weight matrix A and a partitioning in k clusters
% described by the vector v, it computes the cost c associated to the cut
% and the matrix P that describes the partitioning.
% The cost is computed for the double stochastic normalization of A.

    n=size(A,1);
    A=A/(max(sum(A)));
    A=A+sparse(1:n,1:n,1-sum(A));
    P=zeros(n,k);
    for i=1:n
        for j=1:k
            P(i,j)=double(v(i)==j);
        end
    end
    one=ones(n,1);
    c=one'*A*one-trace(P'*A*P);


end