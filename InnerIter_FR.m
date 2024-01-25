function [E,info]=InnerIter_FR(W,epsilon,k,method,Ein)
%% InnerIter_FR: Full Rank inner iteration
% Implements the inner iteration for the function OuterIter_FR.
% Finds the stationary point of the full-rank symmetric ODE
% \dot{E}=-G+<G,E> E. 
% Here G=G(E) is the projection onto the symmetry pattern of
% R=R(E)=Sym(z*one^T-xx^T+yy^T)+c*min(W+epsilon*E,0), 
% where z=x.*x-y.*y and x and y are the normalized eigenvectors of 
% Lap(W+epsilon*E) associated to the k-th and (k+1)-th eigenvalue 
% (ordered increasingly), while one=ones(n,1).
% 
% The stationary point Estar of the ODE determines the optimizer that 
% solves the eigenvalue optimization problem
% argmin_{norm(E,'fro')=1, E=E' and same pattern of W} F_epsilon(E),
% where
% F_epsilon(E)=\lambda_{k+1}(Lap(W+epsilon*E))-\lambda_k(Lap(W+epsilon*E)).
%
% INPUTS
% W: weight matrix of the original graph;
% epsilon: size of the perturbation;
% k: number of clusters to generate;
% method: a structure with fields
%       stepsize (initial stepsize)
%       tol (tolerance)
%       maxit (maximum number of itearations)
%       maxres (maximum number of resets)
%       theta (eventual factor for the decrease and increase of the method)
%       safestop (maximum number of attempts to decrease the functional)
%       pensize (size of the increasing penalization)
%       startpen (starting penalization)
% Ein: initial guess for the starting point Ein of the ODE; (optional)
%
% OUTPUTS
% E: optimal weight perturbation;
% info: a structure with fields
%       derft: time derivative of functional F in the last iteration;
%       derfeps: derivative with respect to epsilon of the functional F;
%       F_path: path of the values of F during the iterations;
%       T-path: path of cumulated times of the iterations;
%       neigs: number of eigs performed.
%
% EXTERNAL FUNCTIONS REQUIRED
% eigtripletks: computation of target eigenvalue and eigenvectors 
% LapSparse: sparse Laplacian of a sparse matrix W Lap(W)=diag(W*one)-W
% projsparse: projection onto the sparsity pattern of a matrix.

    %% SIZE AND PATTERN OF THE WEIGHT MATRIX W
    n=size(W,1);
    [row,col]=find(W);
    
    %% PARAMETERS OF THE METHOD
    h0=method.stepsize;
    tol=method.tol;
    maxit=method.maxit;
    maxres=method.maxres;
    theta=method.theta;
    safestop=method.safestop;
    sigma=method.sigma;
    pen=method.pensize;
    
    %% INIZIALIZATION  
    % original target eigenvalues of the matrix Lap(W)
    [lambda,mu,x,y]=eigtripletks(LapSparse(W),k,sigma);
    f=lambda-mu;
    Ir=[0.25,0,0,0;0,-0.25,0,0;0,0,-1,0;0,0,0,1];
    one=ones(n,1);
    
    % (default) starting value
    if nargin<5
        z=x.*x-y.*y;    
        Uin=[z+one,z-one,x,y];
        Ein=projsparse(row,col,-Ir,Uin);
    end
    E=Ein;
    normE=norm(E,'fro');
    if normE==0
        error('The starting point cannot be the zero matrix.')
    end
    E=E/normE;   
    
    %% STORAGE VECTOR AND INITIALS COUNTERS
    F_path=zeros(maxit,1);
    T_path=zeros(maxit,1);
    F_path(1)=f;
    j=2;
    neigs=2;
    resets=0;
    derft=tol+1;
    c=method.startpen;
    
    %% FIRST ITERATION
    Z=W+epsilon*E;
    [lambda,mu,x,y]=eigtripletks(LapSparse(Z),k,sigma);
    f=lambda-mu+0.5*c*sum(sum(min(Z,0).^2));
    z=x.^2-y.^2;
    U=[z+one,z-one,x,y];
    G=projsparse(row,col,Ir,U)+c*min(Z,0);
             
    %% MAIN COMPUTATION
    while j<maxit && derft>tol && f>tol
        h=h0;
        F_path(j)=f;
        T_path(j)=h;
        
        % Auxiliary quantities for the numerical update
        GscalE=G(:)'*E(:);
        Edot=-G+GscalE*E;

        % Computation of the derivative of the objective function
        normG=norm(G,'fro');
        derf=epsilon*(normG^2-GscalE^2);
        
        % Monotonicity procedure
        fh=f;
        cont=1;
        while fh>=f && cont<safestop && f>tol 
            % Explicit Euler method step with normalization
            Eh=E+h*Edot;
            normEh=norm(Eh,'fro');
            Eh=Eh/normEh;

            % Eigentriplet update
            Wh=W+epsilon*Eh;
            [lambdah,muh,xh,yh]=eigtripletks(LapSparse(Wh),k,sigma);
            neigs=neigs+1;
            fh=lambdah-muh+0.5*c*sum(sum(min(Wh,0).^2));  

            % Eventual reduction of the stepsize
            if fh>=f
                h=h/theta;      
            end
            cont=cont+1;
        end

        % Decide whether the monotonicity procedure failed or not 
        if cont==safestop 
            % Reset the values for the next iteration
            h0=0.1;
            zh=xh.^2-yh.^2;
            U=[zh+one,zh-one,xh,yh];
            G=projsparse(row,col,Ir,U)+c*min(Wh,0);
            f=fh;
            E=Eh;
            resets=resets+1;
        else
            % Eventually reduce the stepsize for the new iteration
            hnext=h;
            if fh>=f-(h/theta)*derf
                hnext=h/theta;
            end

            % Eventually enlarge the stepsize for the new iteration
            if hnext==h0
                ht=h*theta;

                % Explicit Euler method step with normalization
                Et=E+ht*Edot;
                normEt=norm(Et,'fro');
                Et=Et/normEt;

                % Eigentriplet update            
                Wt=W+epsilon*Et;
                [lambdat,mut,xt,yt]=eigtripletks(LapSparse(Wt),k,sigma);
                neigs=neigs+1;
                ft=lambdat-mut+0.5*c*sum(sum(min(Wt,0).^2));

                % Decide whether to enlarge the stepsize or not
                if fh>ft
                    hnext=ht;
                    xh=xt;
                    yh=yt;
                    fh=ft;
                    Eh=Et;
                    Wh=Wt;
                end
            end

            % Update the values for the next iteration
            h0=hnext;
            zh=xh.^2-yh.^2;
            U=[zh+one,zh-one,xh,yh];
            G=projsparse(row,col,Ir,U)+c*min(Wh,0);
            f=fh;
            E=Eh;
        end 
        c=c+pen;
        if resets>=maxres
            f=-1;
        else          
            j=j+1;
        end
    end
    
    %% FINAL OUTPUTS
    F_path=F_path(1:j-1);
    T_path=T_path(1:j-1);
    T_path=cumsum(T_path);
    derfeps=-norm(G,'fro');
    info=struct('derft',derft,'derfeps',derfeps,'F_path',F_path,...
        'T_path',T_path,'neigs',neigs,'c',c); 
    
end