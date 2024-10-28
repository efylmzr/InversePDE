% The format of this file is suitable for describing the conductivity coefficient
% in an elliptic PDE for the assempde.m routine of Matlabs PDE toolbox.
%
% Arguments:
% p     triangulation points
% t     triangle data
% u     not used here
% time  not used here
%
% Returns values of conductivity at centers of mass of triangles.
%
% Calls to: heartNlungs.m
%
% Samuli Siltanen May 2012

function cond = FEMconductivity(p,t,u,time)

% Centers of mass of triangles
xvec = p(1,:);
trix = mean(xvec(t(1:3,:))); 
yvec = p(2,:);
triy = mean(yvec(t(1:3,:))); 

% Evaluate conductivity with auxiliary routine
cond = heartNlungs(trix + i*triy);

