%% Script for the FACEBOOK (Compressed) matrix 

    %% DEFINITION OF THE EXAMPLE
    n=4039;
    nnzhalved=88234;
    M=load('facebook_matrix.txt');
    M=M+ones(nnzhalved,2);
    W=sparse(M(:,1),M(:,2),ones(nnzhalved,1),n,n);
    W=W+W';
    
    % Compression of the model
    m=2;
    k=floor(n/m);
    r=n-rem(n,m);
    W=W(1:r,1:r);
    msquared=m^2;
    rowstart=1;
    B=zeros(k,k);
    for i=1:k
        rowend=rowstart+m-1;
        colstart=1;
        for j=1:i
            colend=colstart+m-1;
            B(i,j)=sum(sum(W(rowstart:rowend,colstart:colend)))/msquared;
            colstart=colend+1;
        end
        rowstart=rowend+1;
    end
    density=length(nonzeros(W))/n^2;
    kdim=k*(k+1)/2;
    nnzB=ceil(density*kdim);
    b=reshape(B,k^2,1);
    [~,index]=sort(b,'descend');
    b(index(nnzB+1:end))=0;
    B=reshape(b,k,k);
    B=(B+B')-diag(diag(B));
    nnzB=2*nnzB-length(nonzeros(diag(B))); 
    n=size(B,1); % n=2019
    W=sparse(B);
    [row,col]=find(W);

    %% PARAMETERS FOR THE ALGORITHMS
    % Inner Iteration
    h=1;
    tol_ii=1e-9;
    maxit=150;
    th=1.3;
    sp=10;
    mr=5;
    pen=0;
    startpen=0;
    sigma=1e-8;
    one=ones(n,1);
    method_ii=struct('integrator','Splitting', 'stepsize',h ,...
        'maxit',maxit, 'maxres', mr, 'theta',th, 'safestop',sp,...
        'sigma',sigma, 'tol',tol_ii, 'pensize',pen, 'startpen',startpen);
    
    % Outer Iteration
    kmin=3;
    kmax=8;
    nk=kmax-kmin+1;
    niter=100;
    lb=1e-7*ones(nk,1);
    ub=10*ones(nk,1);
    tol_out=1e-2;
    
    % Negativity constraint
    tol_neg=1e-5;
    
    %% INIZIALIZATIONS
    d_LR=zeros(nk,1);
    neg_err_LR=zeros(nk,1);
    iter_LR=zeros(nk,1);
    fvec_LR=zeros(nk,1);
    U={[nk,1]};
    S={[nk,1]};
    E_LR={[nk,1]};
    info_oi_LR={[nk,1]};
    d_FR=zeros(nk,1);
    neg_err_FR=zeros(nk,1);
    iter_FR=zeros(nk,1);
    fvec_FR=zeros(nk,1);
    E_FR={[nk,1]};
    info_oi_FR={[nk,1]};
    v=eigs(LapSparse(W),kmax+1,sigma);
    gaps=diff(v(kmin:kmax+1));
    
    %% LOW RANK METHOD
    disp('Low rank method')
    method_oi=struct('el',1e-7, 'eu',10, 'toler',tol_out, 'niter',niter);
    tic;
    j=1;
    for k=kmin:kmax
        disp(['k=',num2str(k),'----------------'])
        method_oi.eu=ub(j);
        [U{j},S{j},info_oi_LR{j}]=OuterIter_LR(W,k,method_ii,method_oi);
        d_LR(j)=info_oi_LR{j}.d(end);
        E_LR{j}=projsparse(row,col,S{j},U{j});
        Z=W+d_LR(j)*E_LR{j};
        Q=Z.*double(Z<0);
        if norm(Q,'fro')>tol_neg
            disp(['Negativity constraint ',num2str(k),'.'])
            Deltastar=(d_LR(j)*E_LR{j}-Q);
            normDeltastar=norm(Deltastar,'fro');
            E_LR{j}=Deltastar/normDeltastar;
            d_LR(j)=normDeltastar;
        end
        neg_err_LR(j)=norm(min(Z,0),'fro');
        iter_LR(j)=info_oi_LR{j}.outiter;
        fvec_LR(j)=info_oi_LR{j}.objfun;
        j=j+1;
    end
    time_LR=toc;
    
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
    disp(['Low rank time: ',num2str(time_LR),' seconds.'])
    disp(['Full rank time: ',num2str(time_FR),' seconds.'])
    close all 
    plot(kmin:kmax,gaps,'c-o')
    hold on
    plot(kmin:kmax,d_LR,'r-o')
    hold on
    plot(kmin:kmax,d_FR,'b-o')
    legend('$g_k(W)$','$d_k^{low}(W)$','$d_k^{full}(W)$',...
        'interpreter','latex')
    title('Comparison between spectral gap and structured distance')   
    