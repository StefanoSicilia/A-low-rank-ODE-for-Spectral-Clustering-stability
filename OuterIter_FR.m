function [E,info]=OuterIter_FR(W,k,method_ii,method_oi)
%% OuterIter_FR: Spectral Graph Clustering Robustness (Full Rank version)
% Given an undirected graph with n-by-n weight matrix W and an integer 
% 0<k<n, it computes the distance, with respect to the Forbenius norm, 
% between W and another weight matrix W_star=W+epsilon_star*E_star such 
% that the graph associated to W_star has vanishing k-th spectral gap.
% Mathematically, by means of spectral clustering, it solves the problem
% argmin_{epsilon>0} F_epsilon(E_star),
% where E_star is the solution of the eigenvalue optimization problem
% argmin_{norm(E,'fro')=1, E=E' and same pattern of W} F_epsilon(E).
% The objective function is
% F_epsilon(E)=\lambda_{k+1}(Lap(W+epsilon*E))-\lambda_k(Lap(W+epsilon*E))
% and \lambda_1(A)<=...<=\lambda_n(A) denote the eigenvalues of the real 
% symmetric n-by-n matrix A.
%
% The algorithm exploits a two-level approach that consists of an inner and
% an outer iteration. The inner iteration is performed by the function
% InnerIter_FR (see this function for more details), while the outer 
% iteration uses a Newton-Bisection method to determine the size of the 
% perturbation epsilon.
%
% INPUTS
% W: weight matrix of the original graph;
% k: number of clusters to generate;
% method_ii: a structure with the inner iteration parameters
% method_oi: a structure with fields
%       el: upper bound for the distance epsilon_star;
%       eu: lower bound for the distance epsilon_star;
%       niter: maximum number of outer iterations;
%       toler: tolerance for the objective fucntion in the outer iteration;
%       sigma (parameter close to 0 to be given to eigs function).
%
% OUTPUTS
% E: the optimizer found for E;
% info: a structure with fields
%       d: the distance found;
%       objfun: the values of the objective function;
%       outiter: number of iterations;
%       neigs: number of eigs performed;
%       dlowerboundeps: lower bound for epsilon;
%       dupperbound: upper bound for epsilon.
%
% EXTRERNAL FUNCTIONS REQUIRED
% InnerIter_FR and all its subfunctions.

    %% CHECK OF THE WEIGHT MATRIX W
    % size and pattern
    [n,p]=size(W);
    [row,col,val]=find(W);
    
    % controls
    if diff([n,p]) 
        error('The weight matrix must be square.') 
    end
    if min(val)<0
        error('The weight matrix must have non-negative entries.')
    end
    if ~isreal(W)
        error('The weight matrix must be real.')
    end
    if ~issymmetric(W)
        error('The weight matrix must be symmetric.')
    end
    if k>=n
        error(['The value of k must be smaller than ',...
            'the size of the weight matrix W.'])
    end
    
    %% PARAMETERS OF THE OUTER ITERATION
    el=method_oi.el;
    eu=method_oi.eu;
    toler=method_oi.toler;
    niter=method_oi.niter;
    sigma=method_ii.sigma;
    
    %% INIZIALIZATION  
    % original target eigenvalues of the matrix Lap(W)
    [~,~,x,y]=eigtripletks(LapSparse(W),k,sigma);
    
    % (default) starting value
    one=ones(n,1);
    z=x.*x-y.*y;
    Ir=[0.25,0,0,0;0,-0.25,0,0;0,0,-1,0;0,0,0,1];
    Uin=[z+one,z-one,x,y];
    [Uin,Sin]=qr(Uin,0);
    Sin=-Sin*Ir*Sin';
    U=Uin;
    S=Sin;
    E=projsparse(row,col,S,U);
    normE=norm(E,'fro');
    if normE==0
        error('The starting point cannot be the zero matrix.')
    end
    epsilon=el;
    neigsout=0;
    j=1;
    
    %% OUTER ITERATIONS
    [E,info_ii]=InnerIter_FR(W,epsilon,k,method_ii,E);
    g=info_ii.derfeps;
    f=info_ii.F_path(end);
    neigsout=neigsout+info_ii.neigs;
    while j<niter && (eu-el)>toler
        if f<toler
            eu=min(eu,epsilon);
            epsilon=0.5*(eu+el);
        else
            el=max(el,epsilon);
            epsilon=epsilon-(f/g);
        end
        if epsilon<el || epsilon>eu
            epsilon=0.5*(el+eu);
        end
        [E,info_ii]=InnerIter_FR(W,epsilon,k,method_ii,E);
        g=info_ii.derfeps;
        f=info_ii.F_path(end);
        neigsout=neigsout+info_ii.neigs;
        j=j+1;
    end
    
    %% OUTPUTS
    disp(['Outer iterations: ',num2str(j),' out of ',num2str(niter),...
        ' for distance ',num2str(epsilon),'.'])
    info=struct('objfun',f,'outiter',j,'d',epsilon,'neigs',neigsout,...
        'dlowerboundeps',el,'dupperbound',eu);

end