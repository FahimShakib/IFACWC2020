%% CL_Main
%--------------------------------------------------------------------------
% Name:     Fahim Shakib
% Program:  Perform the identification of the Logistic map in CLOSED-LOOP
% Date:     31 May 2021
%
% Logistic map state dimension 1, 1 input and 1 output operating in 
% CLOSED-loop with state and output noise (in observer form).
% The noise sequence is estimated using a NARX model with a polynomial
% kernel.
% The state sequence is estimated using the method of Verdult (LS-SVM), 
% using (i) no noise, (ii) the estimated noise sequence and (iii) the true
% noise sequence. Again a polynomial kernel is used. No hyper parameter
% tuning is performed in this steo.
% The mappings f and h are estimated using LS-SVM for aech of the three
% state sequences using again a polynomial kernel.
% The performance is evaluated on the training data and a fresh  validation 
% data set.
%--------------------------------------------------------------------------

%%
clear all;clc

addpath('../')

% SISO
n = 1;
l = 1; % inputs
m = 1; % outputs

% Number of data points
N   = 1000;
% Noise intensity
SNR = 1;
i   = 2;

% Hyper-parameters
v  = 3000;
pp = 2;
cc = 1;

%% Simulation of the true system
% Define random input
r = randn(l,N)*0.1;
t = 0:N-1;

e = 0*r;
% Simulate in Script DT
x0init  = 0;
ynf     = Sim_LM_CL(x0init,r,e*0);
vare    = var(ynf')/(10^(SNR/10));
if any(abs(ynf)>10)
    display('Unstable')
    return
end
e       = sqrt(vare)*randn(m,N);
display(['SNR = ' num2str(10*log10(var(ynf')./var(e')))])

[y0,x0]  = Sim_LM_CL(x0init,r,e);
x0 = x0';
y0 = y0';

u = r-y0;

if any(abs(y0)>10)
    display('Unstable y0')
    return
end

%% Estimate Noise using a NARX model
display(['NARX Estimate'])

hyp      = struct('mean', [], 'cov', [0;0], 'lik', -1); 
covfunc = {@covPoly ,2}; % Second order poly
[eest,f_NARX] = NARX_estimate_NL(y0,u,i,hyp,covfunc);


%% Estimate state with true noise
display(['State Estimate'])

% Estimated Noise
[~,xest_NL,~]         = KCCA_NL(y0(:,2*i+1:end),[u(:,2*i+1:end); eest],         n,i,'poly',[pp cc],v);
% No Noise
[~,xest_NL_nn,~]      = KCCA_NL(y0(:,2*i+1:end),[u(:,2*i+1:end); eest*0],       n,i,'poly',[pp cc],v);
% True Noise
[~,xest_NL_tr,~]      = KCCA_NL(y0(:,2*i+1:end),[u(:,2*i+1:end); e(2*i+1:end)], n,i,'poly',[pp cc],v);
% xest corresponds to x(:,i+1:end-i)

%% Estimate system maps with estimated noise
display(['f and h Estimate'])

% Estimated noise, estimated state 
hyp      = struct('mean', [], 'cov', [0;0], 'lik', -1); 
covfunc = {@covPoly ,2}; % Second order poly
% Estimate Noise
[~,~,f_NL,h_NL]         = fh_estimate(y0(:,3*i+1:end-i),u(:,3*i+1:end-i),eest(:,i+1:end-i),     xest_NL,    covfunc,hyp);
% No Noise
[~,~,f_NL_nn,h_NL_nn]   = fh_estimate(y0(:,3*i+1:end-i),u(:,3*i+1:end-i),eest(:,i+1:end-i)*0,   xest_NL_nn, covfunc,hyp);
% True Noise
[~,~,f_NL_tr,h_NL_tr]   = fh_estimate(y0(:,3*i+1:end-i),u(:,3*i+1:end-i),e(:,3*i+1:end-i),      xest_NL_tr, covfunc,hyp);


%% Validate results on the training dataset
display(['Model Response on Trainings Dataset'])
% Estimated Noise
y_tr_NL     = simulate_ID_NL_model_CL(f_NL,    h_NL,   r,e,0);
% No Noise
y_tr_NL_nn  = simulate_ID_NL_model_CL(f_NL_nn, h_NL_nn,r,e,0);
% True Noise
y_tr_NL_tr  = simulate_ID_NL_model_CL(f_NL_tr, h_NL_tr,r,e,0);

%% Plot and display results training
BFR_tr_NL       = BFR(y0,y_tr_NL);
BFR_tr_NL_nn    = BFR(y0,y_tr_NL_nn);
BFR_tr_NL_tr    = BFR(y0,y_tr_NL_tr);

figure
subplot(211)
plot(y0')
hold all
plot(y_tr_NL','o')
plot(y_tr_NL_nn','x')
plot(y_tr_NL_tr','s')
ylabel('Response')

% Display BFR's
display(['BFR TR NL = ' num2str(BFR_tr_NL) '%'])
display(['BFR TR NL nn = ' num2str(BFR_tr_NL_nn) '%'])
display(['BFR TR NL tr = ' num2str(BFR_tr_NL_tr) '%'])

%% Validate results on a validation dataset
r_val = randn(l,N)*0.1;
e_val = sqrt(vare)*randn(m,N);

% Simulate in Script DT
x0init              = 0;
[y0_val,x0_val]     = Sim_LM_CL(x0init,r_val,e_val);
x0_val              = x0_val';
y0_val              = y0_val';

% Check if response was unstable
if any(abs(y0_val)>10)
    display('Unstable y0_val')
    return
end

display(['Model Response on Validation Dataset'])
% Training Data Set
% Estimated Noise
y_val_NL     = simulate_ID_NL_model_CL(f_NL,   h_NL,    r_val,e_val,0);
% No Noise
y_val_NL_nn  = simulate_ID_NL_model_CL(f_NL_nn,h_NL_nn, r_val,e_val,0);
% True Noise
y_val_NL_tr  = simulate_ID_NL_model_CL(f_NL_tr,h_NL_tr, r_val,e_val,0);

%% Show Validation Results
BFR_val_NL      = BFR(y0_val,y_val_NL);
BFR_val_NL_nn   = BFR(y0_val,y_val_NL_nn);
BFR_val_NL_tr   = BFR(y0_val,y_val_NL_tr);

subplot(212)
plot(y0_val')
hold all
% plot(y_val_LIN','x')
plot(y_val_NL','o')
plot(y_val_NL_nn,'x')
ylabel('Response')

% display(['BFR VAL LIN = ' num2str(BFR_val_LIN) '%'])
display(['BFR VAL NL = ' num2str(BFR_val_NL) '%'])
display(['BFR VAL NL nn = ' num2str(BFR_val_NL_nn) '%'])
display(['BFR VAL NL tr = ' num2str(BFR_val_NL_tr) '%'])

legend('True','est e','no e','true e')