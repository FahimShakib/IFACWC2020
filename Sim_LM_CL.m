%% Sim_LM_CL
%--------------------------------------------------------------------------
% Name:     Fahim Shakib
% Program:  Simulated the logistic map in a closed-loop setting
% Date:     31 May 2021
%--------------------------------------------------------------------------

function [y,x] = Sim_LM_CL(x0,r,e)

% u is single dimensional
% e is single dimensional

f = @(x,u,e)(0.5*x*(1-x)+u+e);

N = length(r);
% Set initial condition
x = x0;

% Simulate identified model
display('Start Simulation')
for k = 1:N
    if not(mod(k,N/50))
        display([num2str(k/N*100) '%'])
    end
    y(k)        = x(1,k) + e(k);
    u           = r(k)-y(k);
    x(:,k+1)    = f(x(:,k),u,e(k));
end

x = x';
y = y';