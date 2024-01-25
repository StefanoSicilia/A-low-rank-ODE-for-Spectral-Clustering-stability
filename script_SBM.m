%% Script for the SBM matrix 

    %% DEFINITION OF THE EXAMPLE
    rng(1)
    ncluster=8;
    sizecluster=20;
    n=ncluster*sizecluster;
    J=rand(sizecluster);
    J=J+J';
    alpha=1;
    B=diag(ones(ncluster-1,1),1)+diag(ones(ncluster-1,1),-1);
    W=sparse(kron(eye(ncluster),J)+kron(B,alpha*eye(sizecluster)));
    [row,col]=find(W);

    %% PARAMETERS FOR THE ALGORITHMS
    % Inner Iteration
    h=1;
    tol_ii=1e-9;
    maxit=150;
    th=1.3;
    sp=10;
    mr=5;
    pen=0.5;
    startpen=0;
    sigma=1e-8;
    one=ones(n,1);
    method_ii=struct('integrator','Splitting', 'stepsize',h ,...
        'maxit',maxit, 'maxres', mr, 'theta',th, 'safestop',sp,...
        'sigma',sigma, 'tol',tol_ii, 'pensize',pen, 'startpen',startpen);
    
    % Outer Iteration
    kmin=3;
    kmax=9;
    nk=kmax-kmin+1;
    niter=100;
    lb=1e-7*ones(nk,1);
    ub=15*ones(nk,1);
    tol_out=1e-2;
    method_oi=struct('el',1e-7, 'eu',15, 'toler',tol_out, 'niter',niter);
    
    % Negativity constraint
    tol_neg=1e-5;
    
    %% INIZIALIZATIONS
    d_FR=zeros(nk,1);
    neg_err_FR=zeros(nk,1);
    iter_FR=zeros(nk,1);
    fvec_FR=zeros(nk,1);
    E_FR={[nk,1]};
    info_oi_FR={[nk,1]};
    v=eigs(LapSparse(W),kmax+1,sigma);
    gaps=diff(v(kmin:kmax+1));
    
    %% FULL RANK METHOD
    disp('Full rank method')
    tic;
    j=1;
    for k=kmin:kmax
        disp(['k=',num2str(k),'----------------']) 
        method_oi.eu=ub(j);
        [E_FR{j},info_oi_FR{j}]=OuterIter_FR(W,k,method_ii,method_oi);
        d_FR(j)=info_oi_FR{j}.d;
        Z=W+d_FR(j)*E_FR{j};
        Q=Z.*double(Z<0);
        if norm(Q,'fro')>tol_neg
            disp(['Negativity constraint ',num2str(k),'.'])
            Deltastar=(d_FR(j)*E_FR{j}-Q);
            normDeltastar=norm(Deltastar,'fro');
            E_FR{j}=Deltastar/normDeltastar;
            d_FR(j)=normDeltastar;
        end
        neg_err_FR(j)=norm(min(Z,0),'fro');
        iter_FR(j)=info_oi_FR{j}.outiter;
        fvec_FR(j)=info_oi_FR{j}.objfun;
        j=j+1;
    end
    time_FR=toc;
    
    %% PLOTS WITH ITERATIONS
    disp(['Full rank time: ',num2str(time_FR),' seconds.'])
    close all 

    figure
    plot(kmin:kmax,gaps,'b-o')
    hold on
    plot(kmin:kmax,d_FR,'r-s')
    legend('$g_k(W)$','$d_k^{full}(W)$','interpreter','latex')
    title('Comparison between spectral gap and structured distance')    
    