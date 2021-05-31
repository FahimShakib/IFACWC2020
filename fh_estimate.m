function [xhat,yhat,f,h] = fh_estimate(y,u,eest,xest,covfunc,hyp)

% Outputs f and h
% f is the prediction of x_{k+1} based on [x_k, u_k, e_k]
% h is the prediction of y_{k}-e_{k} based on x_k

W = [xest(:,1:end-1);u(:,1:end-1);eest(:,1:end-1)];

meanfunc = [];          % Zero prior mean function
likfunc  = @likGauss;     % Gaussian likelihood

% Estimate function f
f_hyp_opt  = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, W', xest(:,2:end)');
% compute response of estimator
% To make predictions using optimal hyperparameters  
[xhat,~] = gp(f_hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, W', xest(:,2:end)', W'); 

f = @(W_test) gp(f_hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, W', xest(:,2:end)', W_test);

% Estimate function h
h_hyp_opt  = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, xest',y'-eest');

% compute response of estimator
% To make predictions using optimal hyperparameters
[yhat,~] = gp(h_hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, xest',y'-eest', xest'); 
yhat = yhat + eest';

h = @(x_test) gp(h_hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, xest', y'-eest', x_test);