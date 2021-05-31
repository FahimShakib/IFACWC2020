%% KCCA_NL
%--------------------------------------------------------------------------
% Name:     Fahim Shakib
% Program:  State estimation for nonlinear system using KCCA
% Date:     31 May 2021
%--------------------------------------------------------------------------

function [ker,xest,kappa] = KCCA_NL(y,u,n,i,ker,kpar,reg)


%% Data
N = length(u);
l = size(u,1);
m = size(y,1);

U = u(:,1:end-2*i);
Y = y(:,1:end-2*i);

% Make Hankel matrices
for k = 1:2*i-1
    utmp = circshift(u,-k,2);
    ytmp = circshift(y,-k,2);
    U = [U;utmp(:,1:end-2*i)];
    Y = [Y;ytmp(:,1:end-2*i)];
end

% Split the data in a part that is from the past and a part that is from
% the future
Up = U(1:i*l,:);
Uf = U(i*l+1:end,:);
Yp = Y(1:i*m,:);
Yf = Y(i*m+1:end,:);
Wp = [Up;Yp];
Wf = [Uf;Yf];


%% Training (finding eta and kappa) using Training Data
[~,~,~,eta,kappa] = ...
    km_kcca(Wf',Wp',ker,kpar,reg,n);

%% Optimize over kernel parameters
% Not done here

%% State Estimate using Training Data
 
% Construct kernels
N0tr    = eye(N-2*i)-1/(N-2*i)*ones(N-2*i);
K_ff    = N0tr*km_kernel(Wf',Wf',ker,kpar)*N0tr;
K_pp    = N0tr*km_kernel(Wp',Wp',ker,kpar)*N0tr;

% State estimate from the past
x_hatp  = kappa'*K_pp;
% State estimate from the future
x_hatf  = eta'*K_ff; 
% Return state estimate from the past (is a bit more reliable)
xest = x_hatp;

