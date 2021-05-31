%% BFR
%--------------------------------------------------------------------------
% Name:     Fahim Shakib
% Program:  Compute the best fit rate
% Date:     31 May 2021
%--------------------------------------------------------------------------

function out = BFR(y0,yest)

out = max(0,(1-norm(y0'-yest',2)/norm(y0-mean(y0,2)))*100);