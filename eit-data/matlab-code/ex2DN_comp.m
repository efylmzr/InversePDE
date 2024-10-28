% Computation of the Dirichlet-to-Neumann map as a matrix acting 
% in trigonometric basis. Omega is the unit disc.
%
% We actually precompute the Neumann-to-Dirichlet map using FEM,
% and then find DN map by inverting the ND map and adding 
% appropriate mapping of constant functions.
%
% We compute in addition the DN map related to the constant 
% conductivity 1 (analytically)
% and save both DN matrices to file data/ex2DN.mat.
% 
% Routine ND_comp.m must be run before this file.
%
% Samuli Siltanen June 2012

% Load precomputed data
load data/ND NtoD Nvec Ntrig
 
% Invert the ND matrix to get DN matrix in trigonometric basis apart
% from constant functions.
DN = inv(NtoD);

% Add appropriate zero row and zero column
DN = [DN(:,1:Ntrig),zeros(2*Ntrig,1),DN(:,Ntrig+1:end)];
DN = [DN(1:Ntrig,:);zeros(1,2*Ntrig+1);DN(Ntrig+1:end,:)];


% Compute DN map of unit conductivity (analytically)
DN1 = abs(diag([-Ntrig:Ntrig]));

% Save result to file
save data/ex2DN DN DN1 Ntrig
