function [eta,RU,nu]=etaRUnu(E,U,x,y,z)
%% etaRUnu: parameters for InnerIter_LR function 
% Computes some variables required in function InnerIter_LR by exploiting 
% sparsity and low rank properties:
% -eta is the scalar <PYR,E>=<R-(I-U*U')*R(I-U*U'),E>;
% -RU is R*U;
% -nu is the scalar <G,E>=<R.*P,E>=<R,E>.
%
% It avoids to store the rank-4 n-by-n matrix
% R=0.5*(z*one'+one*z')-x*x'+y*y', where z=x.^2-y.^2.

    %% INIZIALIZATION
    [n,~]=size(U);
    one=ones(n,1);
    Uone=U'*one;
    Uz=U'*z;    
    Ux=U'*x;
    Uy=U'*y;
    UUone=U*Uone;   Eone=E*one;
    UUz=U*Uz;       Ez=E*z;   
    UUx=U*Ux;       Ex=E*x;
    UUy=U*Uy;       Ey=E*y;        
    
    %% COMPUTATION OF ETA
    eta=(UUone'*Ez+UUz'*Eone)-2*(UUx'*Ex-UUy'*Ey)-...
        0.5*(UUz'*E*UUone+UUone'*E*UUz)+UUx'*E*UUx-UUy'*E*UUy;
    
    %% COMPUTATION OF RU=R*U
    RU=0.5*(one*Uz'+z*Uone')-x*Ux'+y*Uy';    
    
    %% EVENTUAL COMPUTATION OF NU
    if nargout>2
        nu=sum(Ez)-x'*Ex+y'*Ey;
    end
    
end