% Create geometry and FEM mesh for the unit disc
%
% The following information is saved into file data/mesh.mat:
% p          point data matrix
% e          edge data matrix
% t          triangle data matrix
% ecenters   Nx2 matrix of centerpoints of edges
% elengths   lengths of edges
% bfii       Nx1 vector of angles of complex numbers ecenters
% btri_ind   indices of triangles having an edge as one side
%
% Samuli Siltanen March 2012

% Number of mesh refinements. Bigger number giver finer mesh but leads to
% heavier computation as well.
Nrefine = 5;

% Create a subdirectory called 'data'. If it already exists, Matlab will
% show a warning. You don't need to care about the warning.
mkdir('.','data')

% Build Decomposed Geometry Matrix
dgm = [...               % DGM for unit disc
    [ 1, 1, 1, 1];... % 1 = circle domain
    [-1, 0, 1, 0];... % Starting x-coordinate of boundary segment
    [ 0, 1, 0,-1];... % Ending x-coordinate of boundary segment
    [ 0,-1, 0, 1];... % Starting y-coordinate of boundary segment
    [-1, 0, 1, 0];... % Ending y-coordinate of boundary segment
    [ 1, 1, 1, 1];... % Left minimal region label
    [ 0, 0, 0, 0];... % Right minimal region label
    [ 0, 0, 0, 0];... % x-coordinate of the center of the circle
    [ 0, 0, 0, 0];... % y-coordinate of the center of the circle
    [ 1, 1, 1, 1];... % radius of the circle
    [ 0, 0, 0, 0];... % dummy row to match up with the ellipse geometry below
    [ 0, 0, 0, 0];... % dummy row to match up with the ellipse geometry below
    ];

% Construct mesh
[p,e,t] = initmesh(dgm);
for rrr = 1:Nrefine
    [p,e,t] = refinemesh(dgm,p,e,t);
    p       = jigglemesh(p,e,t);
end

% Determine the center points of edges and the corresponding angles
ecenters = (p(:,e(1,:)) + p(:,e(2,:)))/2;
bfii     = angle(ecenters(1,:)+i*ecenters(2,:));

% Find the set of triangles that have a face on the boundary
% and the index of the corresponding edge.
% We use the fact that each triangle in the mesh has zero, one or two
% vertices on the boundary.
btri_ind = zeros(size(bfii));
for ttt = 1:size(t,2) % Loop over all triangles in the mesh
    xtmp = [];
    % Is vertex 1 of current triangle on the bottom?
    if (p(1,t(1,ttt))^2+p(2,t(1,ttt))^2 > 1-1e-10)
        xtmp = [xtmp, p(:,t(1,ttt))];
    end
    % Is vertex 2 of current triangle on the bottom?
    if (p(1,t(2,ttt))^2+p(2,t(2,ttt))^2 > 1-1e-10)
        xtmp = [xtmp, p(:,t(2,ttt))];
    end
    % Is vertex 3 of current triangle on the bottom?
    if (p(1,t(3,ttt))^2+p(2,t(3,ttt))^2 > 1-1e-10)
        xtmp = [xtmp, p(:,t(3,ttt))];
    end
    % At this point, matrix xtmp has at most 2 columns.
    % If the number of columns is two, we have a triangle that has
    % an edge as one side. Let's determine the edge in question.
    if size(xtmp,2)>1
        tmp_center      = (xtmp(1,1)+xtmp(1,2))/2 + i*(xtmp(2,1)+xtmp(2,2))/2;
        tmp_angle       = angle(tmp_center);
        e_ind           = find(abs(bfii-tmp_angle)<1e-8);
        btri_ind(e_ind) = ttt;
    end
end

% Compute lengths of edges
elengths = zeros(size(bfii));
for iii = 1:length(bfii)
    elengths(iii) = norm(p(:,e(1,iii))-p(:,e(2,iii)));
end

% Check the result visually
% figure(1)
% clf
% pdemesh(p,e,t)
% axis equal
% axis off
% print -dpng mesh.png

% Save result to file
save data/mesh p e t bfii ecenters btri_ind elengths dgm

