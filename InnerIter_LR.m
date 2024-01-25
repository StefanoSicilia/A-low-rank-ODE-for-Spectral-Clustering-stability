function [U,S,info]=InnerIter_LR(W,epsilon,k,method,Uin,Sin)
%% InnerIter_LR: Low Rank inner iteration
% Implements the inner iteration for the function OuterIter_LR.
% It finds the stationary point of the rank-4 symmetric ODE
% \dot{Y}=-P_Y(R)+<P_Y(R),E> Y. 
% Here E is the projection of Y=U*S*U^T onto the pattern described by the  
% weight matrix W and R is the low rank gradient
% R=R(E)=Sym(z*one^T-xx^T+yy^T)+c*min(W+epsilon*E,0), 
% where z=x.*x-y.*y and x and y are the normalized eigenvectors of 
% Lap(W+epsilon*E) associated to the k-th and (k+1)-th eigenvalue 
% (ordered increasingly), while one=ones(n,1).
% 
% The stationary point Ystar of the ODE determines the optimizer
% Estar=Ystar.*(W>0) that solves the eigenvalue optimization problem
% argmin_{norm(E,'fro')=1, E=E' and same pattern of W} F_epsilon(E),
% where
% F_epsilon(E)=\lambda_{k+1}(Lap(W+epsilon*E))-\lambda_k(Lap(W+epsilon*E)).
%
% The algorithm updates the matrices U (orthogonal n-by-4) and S 
% (invertible 4-by-4) that characterize the SVD-like decomposition of the 
% rank-4 matrix Y=U*S*U' and solves the system
% \dot{U}=-(I-U*U')*R*S^{-1}
% \dot{S}=-U'*R*U+eta*S
% where eta=<P_Y(R),Y.*P>, which is equivalent to the ODE for Y.
%
% INPUTS
% W: weight matrix of the original graph;
% epsilon: size of the perturbation;
% k: number of clusters to generate;
% method: a structure with fields
%       integrator='Euler' or 'Splitting'
%       stepsize (initial stepsize)
%       maxres (maximum number of resets for imposing monotonicity)
%       tol (tolerance for inner iteration)
%       maxit (maximum number of itearations)
%       theta (factor for the decrease and increase of the stepsize)
%       safestop (maximum number of attempts to decrease the functional)
%       sigma (parameter close to 0 to be given to eigs function);
% Uin: initial guess for the starting point U of the ODE (optional);
% Sin: initial guess for the starting point S of the ODE (optional).
%
% OUTPUTS
% U: stationary point found;
% S: stationary point found;
% info: a structure with fields
%       derft: time derivative of functional F in the last iteration;
%       derfeps: derivative with respect to epsilon of the functional F;
%       F_path: path of the values of F during the iterations;
%       T_path: path of cumulated times of the iterations;
%       neigs: number of eigs performed.
%
% EXTERNAL FUNCTIONS REQUIRED
% eigtripletks: computation of target eigenvalue and eigenvectors;
% projsparse: projection of U*S*U' onto a structural pattern;
% LapSparse: sparse Laplacian of a sparse matrix W (Lap(W)=diag(W*one)-W);
% etaRUnu: computation of some parameters;

    %% SIZE AND PATTERN OF THE WEIGHT MATRIX W
    [n,~]=size(W);
    [row,col]=find(W);
    
    %%  PARAMETERS OF THE METHOD
    integ=method.integrator;
    h0=method.stepsize;
    tol=method.tol;
    maxit=method.maxit;
    maxres=method.maxres;
    theta=method.theta;
    safestop=method.safestop;
    sigma=method.sigma;
    
    %% INIZIALIZATION  
    % original target eigenvalues of the matrix Lap(W)
    [lambda,mu,x,y]=eigtripletks(LapSparse(W),k,sigma);
    f=lambda-mu;
    one=ones(n,1);
    Ir=[0.25,0,0,0;0,-0.25,0,0;0,0,-1,0;0,0,0,1];
    
    % (default) starting values
    if nargin<5
        z=x.*x-y.*y;    
        Uin=[z+one,z-one,x,y];
        [Uin,Sin]=qr(Uin,0);
        Sin=-Sin*Ir*Sin';
    end
    U=Uin;
    S=Sin;
    E=projsparse(row,col,S,U);
    normE=norm(E,'fro');
    if normE==0
        error('The starting point cannot be the zero matrix.')
    end
    E=E/normE;
    S=S/normE;    
    
    %% STORAGE VECTORS AND INITIAL COUNTERS
    F_path=zeros(maxit,1);
    T_path=zeros(maxit,1);
    F_path(1)=f;
    j=2;
    neigs=2;
    resets=0;
    derft=tol+1;
    
    %% FIRST ITERATION
    [lambda,mu,x,y]=eigtripletks(LapSparse(W+epsilon*E),k,sigma);
    f=lambda-mu;
         
    %% MAIN COMPUTATION
    switch integ
        case 'Splitting'
            %% SPLITTING METHOD
            while j<maxit && derft>tol && f>tol
                h=h0;
                % saving results in the storage vectors
                F_path(j)=f;
                T_path(j)=h;
                
                % computation of some coefficients and matrices needed
                z=x.*x-y.*y;
                [eta,RU,GscalE]=etaRUnu(E,U,x,y,z);
                K=U*S;
                Kdot=-RU+eta*K;
                Gamma=[z+one,z-one,x,y];
                G=projsparse(row,col,Ir,Gamma);
                normG=norm(G,'fro');
                derft=epsilon*(normG^2-GscalE^2);

                % Inner cycle to ensure monotonicity of the functional
                fh=f;
                cont=1;
                while fh>=f && cont<safestop 
                    % Update of the step 
                    % i) update of K
                    Kh=K+h*Kdot;
                    [Uh,~]=qr(Kh,0);
                    Mh=Uh'*U;

                    % ii) computation of the new R
                    Sh=Mh*S*Mh';
                    Eh=projsparse(row,col,Sh,Uh);
                    normEh=norm(Eh,'fro');
                    Eh=Eh/normEh;
                    Sh=Sh/normEh;
                    Wh=W+epsilon*Eh;
                    [~,~,xh,yh]=eigtripletks(LapSparse(Wh),k,sigma);
                    neigs=neigs+1;
                    zh=xh.*xh-yh.*yh;
                    [etah,RUh]=etaRUnu(Eh,Uh,xh,yh,zh);

                    % iii) update of S
                    Sh=Sh+h*(-Uh'*RUh+etah*Sh);

                    % iv) eigentriplet update
                    Eh=projsparse(row,col,Sh,Uh);
                    normEh=norm(Eh,'fro');
                    Eh=Eh/normEh;
                    Wh=W+epsilon*Eh;
                    [lambdah,muh,xh,yh]=eigtripletks(LapSparse(Wh),k,sigma);
                    neigs=neigs+1;
                    fh=lambdah-muh; 

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
                    x=xh;
                    y=yh;
                    f=fh;
                    U=Uh;
                    S=Sh;
                    E=Eh;
                    derft=1+tol;
                    resets=resets+1;
                else
                    % Eventually reduce the stepsize for the new iteration
                    hnext=h;
                    if fh>=f-(h/theta)*derft
                        hnext=h/theta;
                    end

                    % Eventually enlarge the stepsize for the new iteration
                    if hnext==h0
                        ht=h*theta;

                        % Update of the step 
                        % i) update of K
                        Kt=K+ht*Kdot;
                        [Ut,~]=qr(Kt,0);
                        Mt=Ut'*U;

                        % ii) computation of the new R
                        St=Mt*S*Mt';
                        Et=projsparse(row,col,St,Ut);
                        normEt=norm(Et,'fro');
                        Et=Et/normEt;
                        St=St/normEt;
                        Wt=W+epsilon*Et;
                        [~,~,xt,yt]=eigtripletks(LapSparse(Wt),k,sigma);
                        neigs=neigs+1;
                        zt=xt.*xt-yt.*yt;
                        [etat,RUt]=etaRUnu(Et,Ut,xt,yt,zt);

                        % iii) update of S
                        St=St+ht*(-Ut'*RUt+etat*St);

                        % Eigentriplet update
                        Et=projsparse(row,col,St,Ut);
                        normEt=norm(Et,'fro');
                        Et=Et/normEt;
                        Wt=W+epsilon*Et;
                        [lambdat,mut,xt,yt]=eigtripletks(LapSparse(Wt),k,sigma);
                        neigs=neigs+1;
                        ft=lambdat-mut;

                        % Decide whether to enlarge the stepsize or not
                        if fh>ft
                            hnext=ht;
                            xh=xt;
                            yh=yt;
                            fh=ft;
                            Uh=Ut;
                            Sh=St;
                            Eh=Et;
                        end
                    end

                    % Update the values for the next iteration
                    h0=hnext;
                    x=xh;
                    y=yh;
                    f=fh;
                    U=Uh;
                    S=Sh;
                    E=Eh;
                end 
                
                % Ensure that S is symmetric
                S=0.5*(S+S');
                
                % Resets check or update
                if resets>=maxres
                    f=-1;
                else          
                    j=j+1;
                end
            end   
        case 'Euler'
            %% EULER METHOD 
            while j<maxit && derft>tol && f>tol
                % saving results in the storage vectors
                F_path(j)=f;
                T_path(j)=h;
                
                % computation of some coefficients and matrices needed
                z=x.*x-y.*y;
                [eta,RU,GscalE]=etaRUnu(E,U,x,y,z);
                URU=U'*RU;
                Udot=(-RU+U*URU)/S;
                Sdot=-URU+eta*S;
                Gamma=[z+one,z-one,x,y];
                G=projsparse(row,col,Ir,Gamma);
                normG=norm(G,'fro');
                derft=epsilon*(normG^2-GscalE^2);

                % Inner cycle to ensure monotonicity of the objective function
                fh=f;
                cont=1;
                while fh>=f && cont<safestop 
                    % Explicit Euler method step with normalization
                    Uh=gs(U+h*Udot); 
                    Sh=S+h*Sdot;
                    Eh=projsparse(row,col,Sh,Uh);
                    normEh=norm(Eh,'fro');
                    Eh=Eh/normEh;
                    Sh=Sh/normEh;

                    % Eigentriplet update
                    Wh=W+epsilon*Eh;
                    [lambdah,muh,xh,yh]=eigtripletks(LapSparse(Wh),k,sigma);
                    neigs=neigs+1;
                    fh=lambdah-muh;  

                    % Eventual reduction of the stepsize
                    if fh>=f
                        h=h/theta;      
                    end
                    cont=cont+1;
                end

                % Decide whether the monotonicity procedure failed or not 
                if cont==safestop 
                    % Reset the values for the next iteration
                    h=0.1;
                    x=xh;
                    y=yh;
                    f=fh;
                    U=Uh;
                    S=Sh;
                    E=Eh;
                    derft=1+tol;
                    resets=resets+1;
                else
                    % Eventually reduce the stepsize for the new iteration
                    hnext=h;
                    if fh>=f-(h/theta)*derft
                        hnext=h/theta;
                    end

                    % Eventually enlarge the stepsize for the new iteration
                    if hnext==h
                        ht=h*theta;

                        % Explicit Euler method step with normalization
                        Ut=gs(U+ht*Udot); 
                        St=S+ht*Sdot;
                        Et=projsparse(row,col,St,Ut);
                        normEt=norm(Et,'fro');
                        Et=Et/normEt;
                        St=St/normEt;

                        % Eigentriplet update
                        Wt=W+epsilon*Et;
                        [lambdat,mut,xt,yt]=eigtripletks(LapSparse(Wt),k,sigma);
                        neigs=neigs+1;
                        ft=lambdat-mut;

                        % Decide whether to enlarge the stepsize or not
                        if fh>ft
                            hnext=ht;
                            xh=xt;
                            yh=yt;
                            fh=ft;
                            Uh=Ut;
                            Sh=St;
                            Eh=Et;
                        end
                    end

                    % Update the values for the next iteration
                    h=hnext;
                    x=xh;
                    y=yh;
                    f=fh;
                    U=Uh;
                    S=Sh;
                    E=Eh;
                end 
                S=0.5*(S+S');
                neigs=neigs+neigs;
                resets=resets+1;
                if resets>=maxres
                    f=-1;
                else          
                    j=j+1;
                end
            end
        otherwise
            %% RAISE ERROR IF METHOD IS NEITHER SPLITTING NOR EULER
            error('Integrator not available.')
    end
    
    
    %% FINAL OUTPUTS
    F_path=F_path(1:j-1);
    T_path=T_path(1:j-1);
    T_path=cumsum(T_path);
    z=x.*x-y.*y;
    Gamma=[z+one,z-one,x,y];
    G=projsparse(row,col,Ir,Gamma);
    derfeps=-norm(G,'fro');
    info=struct('derft',derft,'derfeps',derfeps,'F_path',F_path,...
        'T_path',T_path,'neigs',neigs);    
    
end