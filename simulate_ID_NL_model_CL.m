%% simulate_ID_NL_model_CL
%--------------------------------------------------------------------------
% Name:     Fahim Shakib
% Program:  Simulated identified nonlinear model in a closed-loop setting
% Date:     31 May 2021
%--------------------------------------------------------------------------


function y = simulate_ID_NL_model_CL(f,h,r,e,x0)

% Inputs
% Mappings f,h (Gaussian Process)
% Input r row vector with l rows
% Input e row vector with m rows
% Initial condition X0 in column
%
% Output y

% Extract data length
N = max(size(r));
% Set initial condition
x = x0;
% Simulate identified model
for k = 1:N
    if not(mod(k,N/50))
        display([num2str(k/N*100) '%'])
    end
    y(:,k)  = h(x(:,k)')'+e(:,k);
    u(:,k)  = r(:,k)'-y(:,k)';
    W       = [x(:,k)' u(:,k)' e(:,k)'];
    x(:,k+1)= f(W)';
end
% y = h(x(:,1:end-1)')'+e;