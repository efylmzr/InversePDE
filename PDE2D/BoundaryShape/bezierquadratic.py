import mitsuba as mi 
from mitsuba import Point2f, Bool, Float
from PDE2D.BoundaryShape.interaction import BoundaryInfo
from .boundary_shape import *
from ..utils.helpers import *
from ..utils.sketch import *
from .interaction import *
from ..Coefficient import *
from matplotlib.patches import PathPatch
from matplotlib.path import Path as mpath

class QuadraticBezierShape(Shape):
    ''' Defines the quadratic bezier curve. The neumann boundary part needs to be convex
        otherwise you will get incorrect results. '''
    def __init__(self, vertex_positions : np.array, vertex_normals : np.array = None,
                 dirichlet_map: np.array = None,
                 vertex_dirichlet_vals : list[Float] = None, vertex_neumann_vals : np.array = None, 
                 dirichlet: list[Coefficient] = None,
                 neumann: list[Coefficient] = None,
                 epsilon=1e-4, name = "boundary", normal_derivative_dist = 0.01,
                 inside = True, newton_steps = 5, n_segment = 50):
        super().__init__(np.all(dirichlet_map), single_closed_shape=True, epsilon=epsilon,
                         inside = inside, derivative_dist=normal_derivative_dist)
        self.epsilon_neumann = 1e-5
        self.name = name
        
        self.v_p = Point2f(vertex_positions)
        self.npoints = dr.width(self.v_p)

        if vertex_normals is None:
            indices = dr.arange(UInt32, self.npoints)
            p1 = dr.gather(Point2f, self.v_p, (indices-1) % self.npoints)
            p =  dr.gather(Point2f, self.v_p, (indices) % self.npoints)
            p2 = dr.gather(Point2f, self.v_p, (indices+1) % self.npoints)
            vec1 = p2-p
            vec2 = p1-p
            self.v_n = -dr.normalize(vec1 + vec2) 
        else:
            assert len(vertex_positions) == len(vertex_normals) 
            self.v_n = Point2f(vertex_normals)
        
        dr.make_opaque(self.v_p)
        dr.make_opaque(self.v_n)

        #self.bbox = [[np.min(vertex_positions[0]), np.min(vertex_positions[1])],
        #             [np.max(vertex_positions[0]), np.max(vertex_positions[1])]]
        self.bbox = [[-1, -1],[1,1]]
        self.bbox_center = Point2f(float(self.bbox[0][0] + self.bbox[1][0]) / 2, float(self.bbox[0][1] + self.bbox[1][1]) / 2)
        dr.make_opaque(self.bbox_center)

        if dirichlet_map is None:
            self.dirichlet_map = dr.ones(Bool, self.npoints)
        else:
            self.dirichlet_map = Bool(dirichlet_map)
        dr.make_opaque(self.dirichlet_map)

        self.is_full_dirichlet = dr.all(self.dirichlet_map)
        self.is_full_neumann = dr.all(~self.dirichlet_map)
        self.newton_steps = newton_steps
        self.n_segment = n_segment
        self.NEE = NEE.BruteForce
        self.inside = True

        # Number of segments
        self.n_dirichlet = np.sum(dirichlet_map == True)
        self.n_neumann = np.sum(dirichlet_map == False)

        if not self.is_full_dirichlet:
            if (neumann is None) and (vertex_neumann_vals is None):
                raise Exception("Please specify either a function or vertex neumann values.")
            elif(neumann is None):
                self.num_conf_n = len(vertex_neumann_vals)
                self.v_neumann = dr.zeros(ArrayXf, shape = (self.num_conf_n, self.npoints))
                for i in range(self.num_conf_n):
                    assert len(vertex_neumann_vals[i]) == self.npoints
                    self.v_neumann[i] = Float(vertex_neumann_vals[i])
                self.neumann = None
                dr.make_opaque(self.v_neumann)
            elif vertex_neumann_vals is None:
                self.num_conf_n = len(neumann)
                self.neumann = neumann
                self.v_neumann = None
            else:
                raise Exception("Please only specify either a function, or vertex neumann vals, not both.")

        
        if not self.is_full_neumann:
            if (dirichlet is None) and (vertex_dirichlet_vals is None):
                raise Exception("Please specify either a function or vertex dirichlet values.")
            elif(dirichlet is None):
                self.num_conf_d = len(vertex_dirichlet_vals)
                self.v_dirichlet = dr.zeros(ArrayXf, shape = (self.num_conf_d, self.npoints))
                for i in range(self.num_conf_n):
                    assert len(vertex_dirichlet_vals[i]) == self.npoints
                    self.v_dirichlet[i] = Float(vertex_dirichlet_vals[i])
                self.dirichlet = None
                dr.make_opaque(self.v_dirichlet)
            elif vertex_dirichlet_vals is None:
                self.num_conf_d = len(dirichlet)
                self.dirichlet = dirichlet
                self.v_dirichlet = None
            else:
                raise Exception("Please only specify either a function, or vertex dirichlet vals, not both.")

        if (not self.is_full_dirichlet) and (not self.is_full_neumann):   
            assert (self.num_conf_n == self.num_conf_d) or (self.num_conf_n == 1) or (self.num_conf_d == 1)

        if self.is_full_dirichlet:
            self.num_conf = self.num_conf_d
        elif self.is_full_neumann:
            self.num_conf = self.num_conf_n
        else:
            self.num_conf = max(self.num_conf_d, self.num_conf_n)
        self.max_distance = 10
        self.hasNEE = False
        self.measureCurrent = False

        # Find the control points of the curve using the normals.
        self.c_p = dr.zeros(Point2f, self.npoints)
        for i in range(self.npoints):
            p1 = dr.gather(Point2f, self.v_p, i)
            p2 = dr.gather(Point2f, self.v_p, (i+1) % self.npoints)
            n1 = dr.gather(Point2f, self.v_n, i)
            n2 = dr.gather(Point2f, self.v_n, (i+1) % self.npoints)
            t = (n2[1] * (p2[1]-p1[1]) + n2[0] * (p2[0] - p1[0])) / (n2[1]*n1[0] - n1[1]*n2[0])
            c_p = Point2f(-n1[1] * t + p1[0], n1[0] * t + p1[1])
            dr.scatter(self.c_p, c_p, i)

    def get_opt_params(self, param_dict: dict, opt_params: list):
        pass
    
    def update(self, optimizer):
        pass
    
    def zero_grad(self):
        pass
    
    
    def inside_closed_surface(self, points : Point2f, L : Float, conf_numbers : list[UInt32] = None):
        return self.inside_closed_surface_mask(points), L

    @dr.syntax
    def inside_closed_surface_mask(self, points : Point2f):
        dist_min, bpoint, boundary_normal, k_min, t_min = self.get_closest_dist(points)
        bdir = dr.normalize(bpoint - points)
        threshold = dr.maximum(self.epsilon, 100 * dr.epsilon(Float))
        return (dist_min > threshold) & (dr.dot(boundary_normal, bdir) < 0)

    

    def get_interpolation_points(self, n) -> tuple[Point2f, Point2f, Point2f]:
        p1 = dr.gather(Point2f, self.v_p, n)
        p2 = dr.gather(Point2f, self.v_p, (n+1) % self.npoints)
        c = dr.gather(Point2f, self.c_p, n)
        return p1, p2, c
    
    def get_normals(self, n)-> Point2f:
        return dr.gather(Point2f, self.v_n, n)
    
    def interpolate(self, p1 : Point2f, p2 : Point2f, c : Point2f, t : Float) -> Point2f:
        p1c =  dr.lerp(p1, c, t)
        p2c =  dr.lerp(c, p2, t)
        return dr.lerp(p1c, p2c, t)
    
    def interpolate_derivative(self, p1: Point2f, p2 : Point2f, c : Point2f, t: Float) -> Point2f:
        "Compute derivative with respect to t."
        return -2 * (1 - t) * p1 + 2 * c - 4*c*t + 2 * p2 * t 
    
    def interpolate_derivative2(self, p1 : Point2f, p2 : Point2f, c:Point2f, t:Float) -> Point2f:
        return 2 * p1 + 2 * p2 - 4 * c

    def get_distance2(self, p : Point2f, p1 : Point2f, p2 : Point2f, c : Point2f, t : Float) -> tuple[Float, Float]:
        "Compute the distance squared to a given t and derivative and 2nd derivative of it."
        p_t = self.interpolate(p1, p2, c, t)
        p_t_der = self.interpolate_derivative(p1, p2, c, t)
        p_t_der2 = self.interpolate_derivative2(p1, p2, c, t)
        p_diff = p - p_t 
        dist2 = dr.squared_norm(p_diff)
        dist2_der = -2 * p_diff * p_t_der
        dist2_der = dist2_der[0] + dist2_der[1]
        dist2_der2 = -2 * p_diff * p_t_der2 + 2 * dr.square(p_t_der)
        dist2_der2 = dist2_der2[0] + dist2_der2[1]
        return dist2, dist2_der, dist2_der2
    
    def min_segment(self, p : Point2f, p1 : Point2f, p2 : Point2f, c : Point2f, i_segment : UInt32, n_segment : UInt32) -> Float:
        t1 = Float(i_segment) / n_segment
        t2 = Float(i_segment + 1) / n_segment

        pl1 = self.interpolate(p1, p2, c, t1)
        pl2 = self.interpolate(p1, p2, c, t2)
        vec_edge = pl2 - pl1
        t = dr.clamp(dr.dot((p - pl1), vec_edge) / dr.squared_norm(vec_edge), 0, 1)
        min_point = dr.lerp(pl1, pl2, t)
        return dr.norm(min_point - p), t / n_segment + t1
    
    @dr.syntax
    def get_closest_dist_polyline_k(self, p : Point2f, k : UInt32):
        p1, p2, c = self.get_interpolation_points(k)
        # First find the intersection point if it was linearly interpolated.
        i = UInt32(0)
        dist_min = Float(dr.inf)
        t = Float(1)
        i_selected = UInt32(0)
        while i < self.n_segment:
            min, t_ = self.min_segment(p, p1, p2, c, i, self.n_segment)
            if min < dist_min:
                dist_min = min
                t = t_
                i_selected = UInt32(i)
            i += 1
        return dist_min, t, i_selected
    
    @dr.syntax
    def get_closest_dist_polyline(self, p : Point2f):
        i = UInt32(0)
        dist_min = Float(dr.inf)
        n_min = UInt32(0)
        t_min = Float(dr.inf)

        while i < self.npoints:
            dist, t, _ = self.get_closest_dist_polyline_k(p, i)
            if dist < dist_min:
                n_min = i
                dist_min = dist
                t_min = t
            i+=1
        return dist_min, n_min, t_min
    
    @dr.syntax
    def get_closest_dist_k(self, p : Point2f, k : UInt32):
        dist_min, t_min, i_min = self.get_closest_dist_polyline_k(p, k)
        p1, p2, c = self.get_interpolation_points(k)

        
        a1, a2 = Float(i_min) / self.n_segment, Float(i_min + 1) / self.n_segment
        # Start to apply Newton's algorithm to set the derivative to zero.
        i = UInt32(0)
        dist2 = Float(dr.inf)
        while i < self.newton_steps:
            dist2, val, deriv = self.get_distance2(p, p1, p2, c, t_min)
            dist_min = dr.sqrt(dist2)
            t_min = t_min - val / deriv

            # Newton-Bisection: potentially reject the Newton step
            bad_step = ~((t_min >= a1) & (t_min <= a2))
            t_min = dr.select(bad_step, (a1 + a2) / 2, t_min)

            # Update bracketing interval
            is_neg = self.get_distance2(p, p1, p2, c, t_min)[1]  < 0
            a1 = dr.select(is_neg, t_min, a1)
            a2 = dr.select(is_neg, a2, t_min)

            t_min = dr.clamp(t_min, 0, 1)
            i += 1
        bpoint = self.interpolate(p1, p2, c, t_min)

        boundary_normal = self.interpolate_derivative(p1, p2, c, t_min)
        boundary_normal = dr.normalize(Point2f(-boundary_normal[1], boundary_normal[0]))

        aprox_n1 = self.get_normals(k)
        aprox_n2 = self.get_normals((k + 1) % self.npoints)
        aprox_n = dr.lerp(aprox_n1, aprox_n2, t_min)
        if dr.dot(aprox_n, boundary_normal) < 0:
            boundary_normal = -boundary_normal

        if dr.hint(self.inside, mode = "scalar"):
            boundary_normal = -boundary_normal

        return dist_min, bpoint, boundary_normal, t_min

    
    @dr.syntax
    def get_closest_dist(self, p : Point2f):
        # First get the closest point if it was linearly interpolated and find t.
        dist_min, k_min, t_min = self.get_closest_dist_polyline(p)
        p1, p2, c = self.get_interpolation_points(k_min)
        
        # Start to apply Newton's algorithm to set the derivative to zero.
        i = UInt32(0)
        dist2 = Float(dr.inf)
        while i < self.newton_steps:
            dist2, val, deriv = self.get_distance2(p, p1, p2, c, t_min)
            dist_min = dr.sqrt(dist2)
            t_min = t_min - val / deriv
            t_min = dr.clamp(t_min, 0, 1)
            # if t is smaller than 0 or greater than one, it means we changed the segment.
            if t_min<0:
                k_min = (k_min-1) % self.npoints
                t_min = Float(1)
                p1, p2, c = self.get_interpolation_points(k_min)
            elif t_min>1:
                k_min = (k_min+1) % self.npoints
                t_min = Float(0)
                p1, p2, c = self.get_interpolation_points(k_min)
            i += 1
        bpoint = self.interpolate(p1, p2, c, t_min)
        boundary_normal = self.interpolate_derivative(p1, p2, c, t_min)
        boundary_normal = dr.normalize(Point2f(-boundary_normal[1], boundary_normal[0]))

        aprox_n1 = self.get_normals(k_min)
        aprox_n2 = self.get_normals((k_min + 1) % self.npoints)
        aprox_n = dr.lerp(aprox_n1, aprox_n2, t_min)
        if dr.dot(aprox_n, boundary_normal) < 0:
            boundary_normal = -boundary_normal

        if dr.hint(self.inside, mode = "scalar"):
            boundary_normal = -boundary_normal
        return dist_min, bpoint, boundary_normal, k_min, t_min
    
    @dr.syntax
    def boundary_interaction(self, points: Point2f, radius_fnc: callable = None, 
                             max_radius=Float(dr.inf) , star_generation=True, 
                             conf_numbers : list[UInt32] = [UInt32(0)]) -> BoundaryInfo:
        if dr.hint(self.is_full_dirichlet, mode = "scalar"):
            closest_dist, closest_bpoint, closest_bnormal, k_all, t_min_all = self.get_closest_dist(points)
            k_dirichlet, t_min_dirichlet = Float(k_all), Float(t_min_all)
            closest_bpoint_dirichlet = closest_bpoint
            closest_dirichlet = closest_dist
            is_dirichlet = Bool(True)
            is_neumann = Bool(False)
        elif dr.hint(self.is_full_neumann, mode = "scalar"):
            closest_dist, closest_bpoint, closest_bnormal, k_all, t_min_all = self.get_closest_dist(points)
            closest_dirichlet = Point2f(dr.inf)
            closest_bpoint_dirichlet = Point2f(dr.nan)
            is_dirichlet = Bool(False)
            is_neumann = Bool(True)
        else:
            is_dirichlet = Bool(True)
            i = UInt32(0)
            k_dirichlet = UInt32(0)
            k_all = UInt32(0)
            closest_dist = Float(dr.inf)
            closest_dirichlet = Float(dr.inf)
            closest_bpoint = Point2f(dr.nan)
            closest_bpoint_dirichlet = Point2f(dr.nan)
            closest_bnormal = Point2f(dr.nan)
            t_min_dirichlet = Float(dr.inf)
            t_min_all = Float(dr.inf)
            while i < self.npoints:
                d, bpoint, bnormal, t = self.get_closest_dist_k(points, i)
                if d<closest_dist:
                    closest_dist = d
                    closest_bpoint = bpoint
                    closest_bnormal = bnormal
                    k_all = i
                    t_min_all = t
                    is_dirichlet = dr.gather(Bool, self.dirichlet_map, i)
                if (d < closest_dirichlet) &  dr.gather(Bool, self.dirichlet_map, i):
                    closest_dirichlet = d
                    closest_bpoint_dirichlet = bpoint
                    t_min_dirichlet = t
                    k_dirichlet = i

                #if closest_dist < closest_dirichlet:
                #    is_dirichlet = Bool(False)
                i += 1
                
            merge_dirichlet = (closest_dirichlet < (100*dr.epsilon(Float))) 
            is_dirichlet |=  merge_dirichlet 
            is_neumann = ~is_dirichlet 
            
        num_conf = len(conf_numbers)
        nearest_dirichlet_val = dr.zeros(ArrayXf, shape = (num_conf, dr.width(points)))
        if dr.hint(not self.is_full_neumann, mode = "scalar"):
            if dr.hint(self.v_dirichlet is None, mode = "scalar"):    
                    if dr.hint(self.num_conf_d == 1, mode = 'scalar'):
                        dirichlet = self.dirichlet[0].get_value(closest_bpoint_dirichlet)
                        for i in range(num_conf):
                            nearest_dirichlet_val[i] = dirichlet
                    else:
                        for i in range(self.num_conf_d): 
                            for j, conf in enumerate(conf_numbers):
                                if i == conf:
                                    nearest_dirichlet_val[j] = self.dirichlet[i].get_value(closest_bpoint_dirichlet)
            else:
                nearest_dirichlet_all = self.get_dirichlet_vertices(k_dirichlet, t_min_dirichlet)
                if dr.hint(self.num_conf_d == 1, mode = 'scalar'):
                    nearest_dirichlet_val = nearest_dirichlet_all
                else:
                    for i in range(self.num_conf_d): 
                            for j, conf in enumerate(conf_numbers):
                                if i == conf:
                                    nearest_dirichlet_val[j] = nearest_dirichlet_all[i]
        
        
        boundary_dir = dr.normalize(closest_bpoint - points) 
        radius = closest_dirichlet if radius_fnc is None else radius_fnc(closest_dirichlet)
        radius = dr.minimum(radius, max_radius)
        
        is_epsilon_shell = ((closest_dist < self.epsilon)) 
        on_boundary = (closest_dist < self.epsilon_neumann)
        is_far = (self.inf_distance < closest_dist) & (dr.dot(boundary_dir, closest_bnormal) > 0)
        
        # Curvature computation
        # We computed some stuff here, but whatever.
        p1, p2, c = self.get_interpolation_points(k_all)
        pd = self.interpolate_derivative(p1, p2, c, t_min_all)
        pd2 = self.interpolate_derivative2(p1, p2, c, t_min_all)
        pd_sqr = dr.sum(dr.square(pd))
        curvature =  -(pd[0] * pd2[1] - pd[1] * pd2[0]) / dr.sqrt(dr.square(pd_sqr) * pd_sqr)
        bi = BoundaryInfo(points, on_boundary, radius, closest_dist, is_far, closest_bpoint, curvature, closest_dirichlet, nearest_dirichlet_val, 
                            closest_bpoint_dirichlet, boundary_dir, closest_bnormal, is_dirichlet, is_neumann, is_epsilon_shell, 
                            UInt32(0), UInt32(0), is_star = ~is_dirichlet)
        return bi
    
    def star_generation(self, bi):
        bi.is_star = bi.is_n & (bi.r > bi.d)
        return bi
    
    @dr.syntax
    def ray_intersect(self, bi : BoundaryInfo, direction : Point2f, conf_numbers : list[UInt32] = None):

        origin = Point2f(bi.origin)
        if bi.on_boundary:
            origin = Point2f(bi.bpoint)  + bi.bn * self.epsilon_neumann/50

        #o_b = 1/bi.curvature * bi.bn
        #cos_angle = dr.dot(o_b, direction)
        #aprox_mask = (bi.curvature > 0) & bi.on_boundary & cos_angle < 1e-2
        #if aprox_mask:
        #    t_min = 2 * cos_angle / bi.curvature

        epsilon = 0
        i = UInt32(0)
        # First find which ray segment we should search for by assuming it is linearly interpolated.
        n = UInt32(0)
        t_min = Float(dr.inf)
        t_min_ = Float(dr.nan)
        while i < self.npoints:
            p1, p2, co = self.get_interpolation_points(i)
            a_ = p1 + p2 - 2 * co
            b_ = 2 * (co - p1)
            c_ = p1 - origin
            a = direction[0] * a_[1] - direction[1] * a_[0]
            b = direction[0] * b_[1] - direction[1] * b_[0]
            c = direction[0] * c_[1] - direction[1] * c_[0]
            t1_ = (-b  - dr.sqrt(dr.square(b) - 4 * a * c)) / (2 * a)
            t2_ = (-b  + dr.sqrt(dr.square(b) - 4 * a * c)) / (2 * a)
            
            t1 = Float(0)
            t2 = Float(0)
            if dr.abs(direction[0]) < 0.7:
                t1 = (a_[1] * dr.square(t1_) + b_[1] * t1_ + c_[1]) / direction[1]
                t2 = (a_[1] * dr.square(t2_) + b_[1] * t2_ + c_[1]) / direction[1]
            else:
                t1 = (a_[0] * dr.square(t1_) + b_[0] * t1_ + c_[0]) / direction[0]
                t2 = (a_[0] * dr.square(t2_) + b_[0] * t2_ + c_[0]) / direction[0]
            if dr.isfinite(t1) & dr.isfinite(t1_) & (t1 < t_min) & (t1> epsilon) & (t1_ >= 0) & (t1_ <= 1):
                t_min = t1
                t_min_ = t1_
                n = UInt32(i)
            if dr.isfinite(t2) & dr.isfinite(t2_) & (t2 < t_min) & (t2> epsilon) & (t2_ >= 0) & (t2_ <= 1):
                t_min = t2
                t_min_ = t2_
                n = UInt32(i)
            i+=1
        #if self.inside & (t_min == dr.inf) & bi.on_boundary:
        #    t_min = Float(0)

        intersected = origin + direction * t_min
        p1, p2, co = self.get_interpolation_points(n)
        normal = self.interpolate_derivative(p1, p2, co, t_min_)
        normal = Point2f(-normal[1], normal[0])
        if dr.dot(direction, normal) > 0:
            normal = -normal

        # Get the boundary condition at the hit point.
        is_dirichlet = dr.gather(Bool, self.dirichlet_map, n)

        neumann_vals = None
        if dr.hint(conf_numbers is not None, mode = 'scalar'):
            num_conf = len(conf_numbers)
            neumann_vals = dr.zeros(ArrayXf, shape = (num_conf, dr.width(bi.origin)))
            if not self.is_full_dirichlet:
                if dr.hint(self.v_neumann is None, mode = "scalar"):    
                    if dr.hint(self.num_conf_n == 1, mode = 'scalar'):
                        neumann = self.neumann[0].get_value(intersected)
                        for i in range(num_conf):
                            if ~is_dirichlet:
                                neumann_vals[i] = neumann
                    else:
                        for i in range(self.num_conf_n): 
                            for j, conf in enumerate(conf_numbers):
                                if (i == conf) & ~is_dirichlet:
                                    neumann_vals[j] = self.neumann[i].get_value(intersected)
                else:
                    neumann_all = self.get_neumann_vertices(n, t_min_)
                    if dr.hint(self.num_conf_n == 1, mode = 'scalar'):
                        for i in range(num_conf):
                            if ~is_dirichlet:
                                neumann_vals = Float(neumann_all)
                    else:
                        for i in range(self.num_conf_n): 
                            for j, conf in enumerate(conf_numbers):
                                if i == conf & ~is_dirichlet:
                                    neumann_vals[j] = Float(neumann_all[i])

        ri = RayInfo(origin, direction, t_min, intersected, normal, is_dirichlet, neumann_vals)
        return ri
    
    def get_diriclet_value(self, points) -> Float:
        self.get_closest_dist(points,)
        
    def sketch(self, ax, bbox, resolution, colors = ["red", "green"], sketch_center = False, sketch_in_boundaries = False, lw = None):
        for i in range(self.npoints):
            color = colors[0] if self.dirichlet_map[i]==True else colors[1]
            p1, p2, c = self.get_interpolation_points(i)
            p1_s = point2sketch(p1, bbox, resolution).numpy().squeeze()
            p2_s = point2sketch(p2, bbox, resolution).numpy().squeeze()
            c_s = point2sketch(c, bbox, resolution).numpy().squeeze()
            pp = PathPatch(mpath([p1_s, c_s, p2_s], [mpath.MOVETO, mpath.CURVE3, mpath.CURVE3]), facecolor='none', edgecolor = color, linewidth = lw, capstyle = "round")
            ax.add_patch(pp)

    def sketch_points(self, ax, bbox, resolution, colors = ['green', 'orange'], control_points : bool = False):
        for i in range(self.npoints):
            p1, p2, c = self.get_interpolation_points(i)
            p1_s = point2sketch(p1, bbox, resolution).numpy().squeeze()
            p2_s = point2sketch(p2, bbox, resolution).numpy().squeeze()
            c_s = point2sketch(c, bbox, resolution).numpy().squeeze()
            ax.scatter(p1_s[0], p1_s[1], color = colors[0])
            if control_points:
                ax.scatter(c_s[0], c_s[1], color = colors[1])

    def sketch_curvature(self, bi : BoundaryInfo, ax, bbox, resolution):
        radius = 1/bi.curvature
        center = bi.bpoint + bi.bn * radius
        bpoint_s = point2sketch(bi.bpoint, bbox, resolution).numpy()
        center_s = point2sketch(center, bbox, resolution).numpy()
        radius_s = dist2sketch(radius, bbox, resolution)[0].numpy()
        ax.scatter(bpoint_s[0], bpoint_s[1], color = 'red')
        ax.scatter(center_s[0], center_s[1])
        for c, r in zip(center_s.T, radius_s):
            sphere = patches.Ellipse(c, r * 2, r * 2, linewidth= 1,
                                fill = False, color = "purple")
            ax.add_patch(sphere)        


    def sketch_normals(self, ax, bbox, resolution, color = "blue", length_vector = 0.2):
        for i in range(self.npoints):
            p1, _, c = self.get_interpolation_points(i)
            p1_s = point2sketch(p1, bbox, resolution).numpy().squeeze()
            c_s = point2sketch(c, bbox, resolution).numpy().squeeze()
            l_s, _, _ = dist2sketch(length_vector, bbox, resolution)
            v_n = self.get_normals(i)
            n_s = dir2sketch(v_n, bbox, resolution).numpy().squeeze()
            arrow = patches.FancyArrow(p1_s[0], p1_s[1], n_s[0] * l_s, n_s[1] * l_s, 
                                        width = 2 / 512 * resolution[0], length_includes_head=True, color = color)
            ax.add_patch(arrow)


    def sketch_polyline(self, ax, bbox, resolution, color = "blue"):
        for i in range(self.npoints):
            p1, p2, c = self.get_interpolation_points(i)
            t = dr.arange(Float, self.n_segment + 1) / self.n_segment
            p = self.interpolate(p1, p2, c, t)
            p_s = point2sketch(p, bbox, resolution).numpy().squeeze()
            pp = PathPatch(mpath(p_s.T), facecolor='none', edgecolor = color)
            ax.add_patch(pp)


    def get_dirichlet_vertices(self, k : UInt32, t :UInt32):
        d1 = dr.gather(Float, self.v_dirichlet, k)
        d2 = dr.gather(Float, self.v_dirichlet, (k + 1) % self.npoints)
        return dr.lerp(d1, d2, t)
    
    def get_neumann_vertices(self, k : UInt32, t :UInt32):
        n1 = dr.gather(Float, self.v_neumann, k)
        n2 = dr.gather(Float, self.v_neumann, (k + 1) % self.npoints)
        return dr.lerp(n1, n2, t)
    
    @dr.syntax
    def create_neumann_points(self, resolution : int, spp : int):
        n = resolution * spp
        points = dr.zeros(Point2f, self.n_neumann * n)
        sampler = PCG32(initstate=dr.arange(UInt32, n))
        indices = dr.arange(UInt32, n)
        j = UInt32(0)
        for i in range(self.npoints):
            is_d = dr.gather(Bool, self.dirichlet_map, i)
            if ~is_d:
                p1, p2, c = self.get_interpolation_points(i)
                t = dr.arange(Float, n) / n + sampler.next_float32() / n
                p_iter = self.interpolate(p1, p2, c, t)
                dr.scatter(points, p_iter, indices + j)
                j += n
        return points
    
    def create_boundary_points(self, resolution : int, spp : int):
        n = resolution * spp
        points = dr.zeros(Point2f, self.n_segment * n)
        sampler = PCG32(initstate=dr.arange(UInt32, n))
        indices = dr.arange(UInt32, n)
        j = UInt32(0)
        for i in range(self.npoints):
            p1, p2, c = self.get_interpolation_points(i)
            t = dr.arange(Float, n) / n + sampler.next_float32() / n
            p_iter = self.interpolate(p1, p2, c, t)
            dr.scatter(points, p_iter, indices + j)
            j += n
        return points



    
        

            






