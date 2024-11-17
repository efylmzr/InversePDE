% Compute the Neumann-to-Dirichlet map of the conductivity heartNlungs.m.
%
% Samuli Siltanen June 2012

% Load mesh and decomposed geometry matrix
% (precomputed with the routine mesh_comp.m)
load data/mesh p e t dgm

% Order of trigonometric approximation
Ntrig = 16;

% Build NtoD matrix element by element
Nvec  = [[-Ntrig : -1],[1 : Ntrig]];
NtoD  = zeros(length(Nvec));
for nnn = 1:length(Nvec)
    
    % Power of trigonometric basis function used as boundary data. 
    % We save n to disc, and it will be loaded by function BoundaryData.m.
    n = Nvec(nnn); 
    save data/BoundaryDataN n
        
    % Solve elliptic PDE with FEM
    u = assempde('BoundaryData',p,e,t,'FEMconductivity',0,0);
    
    %figure(1)
    %clf
    %pdesurf(p,t,real(u))
    %drawnow
    %pause
    
    % Compute trace of solution
    Nfii = size(e,2);
    fii  = zeros(Nfii,1);
    u_tr = zeros(Nfii,1);
    for iii = 1:Nfii
        % We use the fact that in mesh_comp.m the unit circle was 
        % divided into the following four segments: 
        % (1) [pi,3*pi/2], (2) [3*pi/2,0], (3) [0,pi/2], (4) [pi/2,pi]
        fii(iii) = pi + (e(5,iii)-1)*pi/2 + e(3,iii)*pi/2;
        % Now we pick the corresponding values of the trace
        u_tr(iii) = u(e(1,iii));
    end
    
    % Sort the angles and arrange the corresponding values accordingly
    [fii,ind] = sort(fii);
    u_tr      = u_tr(ind);
    
    % Expand the traces in trigonometric basis. Here we assume that the angles
    % fii are equidistant
    Dfii = fii(2)-fii(1);
    for jjj = 1:length(Nvec)
        NtoD(jjj,nnn) = 1/sqrt(2*pi)*Dfii*exp(i*Nvec(jjj)*fii)'*u_tr;
    end 
    disp(['Done ', num2str(nnn), ' out of ', num2str(length(Nvec))])
end % for nnn

% Save result to file
save data/ND NtoD Nvec Ntrig
 
 
 