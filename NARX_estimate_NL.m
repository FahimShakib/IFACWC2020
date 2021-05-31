%% NARX_estimate_NL
%--------------------------------------------------------------------------
% Name:     Fahim Shakib
% Program:  Estimate a NARX model using the GPML toolbox
% Date:     31 May 2021
%--------------------------------------------------------------------------

function [e,NARX] = NARX_estimate_NL(y,u,i,hyp,covfunc)
% i is the stacking length

% Take last samples of y

N = length(u);
l = size(u,1);
m = size(y,1);

U = u(:,1:end-2*i);
Y = y(:,1:end-2*i);

for k = 1:2*i-1
    utmp = circshift(u,-k,2);
    ytmp = circshift(y,-k,2);
    U = [U;utmp(:,1:end-2*i)];
    Y = [Y;ytmp(:,1:end-2*i)];
end

W = [U;Y];

meanfunc = [];          % Zero prior mean function
% hyp      = struct('mean', [], 'cov', 0, 'lik', -1); 
% covfunc  = @covLINiso;    % Squared Exponental covariance function
likfunc  = @likGauss;        % Gaussian likelihood
hyp_opt  = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, W', y(:,2*i+1:end)');

% compute response of estimator
% To make predictions using optimal hyperparameters  
[yhat,~] = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, W', y(:,2*i+1:end)', W'); 
e = y(:,2*i+1:end) - yhat';

NARX = @(W_test) gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, W', y(:,2*i+1:end)', W_test); 