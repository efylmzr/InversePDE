from .boundary_shape import *
from ..utils.helpers import *
from ..utils.sketch import *
from .interaction import *
from mitsuba import UInt
from ..utils.imageUtils import create_circle_from_result, create_circle_points
from ..Coefficient import *
from .sdf_grid import SDFGrid
class CircleShape(Shape):
    def __init__(self, origin=[0, 0], radius=1,
                 angle_partition=np.array([0]),
                 dirichlet_map: np.array = np.array([True]),
                 dirichlet: list[Coefficient] = [ConstantCoefficient("dirichlet", 0)],
                 neumann: list[Coefficient] = [ConstantCoefficient("neumann", 0)],
                 epsilon=1e-4, name = "boundary", normal_derivative_dist = 0.01,
                 inside = True):
        super().__init__(np.all(dirichlet_map), single_closed_shape=True, epsilon=epsilon,
                         inside = inside, derivative_dist=normal_derivative_dist)
        self.name = name
        self.origin = Point2f(origin)
        self.radius = Float(radius)
        dr.make_opaque(self.origin)
        dr.make_opaque(self.radius)
        # check if there is no neumann boundary
        self.dirichlet_map = Bool(dirichlet_map)
        dr.make_opaque(self.dirichlet_map)
        self.is_full_dirichlet = dr.all(self.dirichlet_map)
        self.is_full_neumann = dr.all(~self.dirichlet_map)

        # Create interval vector of size (n_interval, 2)
        # The first angle partition
        self.init_angle_partition = Float(float(angle_partition[0]))
        angle_partition = np.append(
            angle_partition, angle_partition[0] + 2 * np.pi)
        angle_intervals = np.stack([angle_partition[:-1],
                                    angle_partition[1:]], axis=0)
        self.angle_partition = Point2f(angle_intervals)
        dr.make_opaque(angle_partition)
        self.num_intervals = dr.width(self.angle_partition)

        # Get the dirichlet and neumann intervals
        self.dirichlet_angles = Point2f(angle_intervals[:, np.array(dirichlet_map, dtype = bool)])
        self.neumann_angles = Point2f(angle_intervals[:, ~np.array(dirichlet_map, dtype = bool)])

        self.bbox = [[self.origin[0] - radius, self.origin[1] - radius], 
                     [self.origin[0] + radius, self.origin[1] + radius]]
    
        # Dirichlet and neumann boundary values (instance of Coefficient).
        self.dirichlet = dirichlet
        self.neumann = neumann

        self.num_conf_d = len(dirichlet)
        self.num_conf_n = len(neumann)
        assert (self.num_conf_d == self.num_conf_n) or (self.num_conf_n == 1) or (self.num_conf_d == 1)

        self.max_distance = self.radius
        self.measureCurrent = False

    @dr.syntax
    def ray_intersect(self, bi : BoundaryInfo, direction : Point2f, conf_numbers : list[UInt32] = None):  # change this for outside!

        # if outside due to numerical errors (we always assume that we compute solution inside!)
        # origin = dr.select(dr.norm(o_b) >= self.radius - 1e-1,
        #                   self.origin - o_b * (self.radius - 1e-1) / dr.norm(o_b),
        #                   origin)
        # Solve second order polynomial
        origin = Point2f(bi.origin)
        on_boundary = Bool(bi.on_boundary)
        o_b = Point2f(0)
        if on_boundary & self.inside:
            o_b = dr.normalize(self.origin - origin)
            cos_angle = dr.dot(o_b, direction)
            t = 2 * cos_angle * self.radius
        else:
            o_b = origin - self.origin
            a = dr.dot(direction, direction)
            b = 2 * dr.dot(o_b, direction)
            c = dr.dot(o_b, o_b) - dr.sqr(self.radius)
            sign = dr.select(self.inside, 1, -1)
            t = (- b + sign * dr.sqrt(dr.sqr(b) - 4 * a * c)) / (2 * a)

        intersected = mi.Point2f(origin + direction * t)
        # if intersected point is outside the domain, put it back in!
        diff = intersected -self.origin
        normals = dr.normalize(diff)
        normals = -normals if self.inside else normals

        angles = correct_angle(dr.atan2(diff[0], diff[1]))
        is_dirichlet = self.is_dirichlet_boundary(angles, Bool(True))
        
        neumann_vals = None
        if dr.hint(conf_numbers is not None, mode = 'scalar'):
            num_conf = len(conf_numbers)
            neumann_vals = dr.zeros(ArrayXf, shape = (num_conf, dr.width(intersected))) 
            if not dr.hint(self.is_full_dirichlet, mode = "scalar"):
                if ~is_dirichlet:
                    if dr.hint(self.num_conf_n == 1, mode = "scalar"):
                        neumann = self.neumann[0].get_value(intersected)
                        for i in range(num_conf):
                            neumann_vals[i] = neumann
                    else:
                        for i in range(self.num_conf_n): 
                            for j, conf in enumerate(conf_numbers):    
                                neumann_vals[j] = dr.select(i == conf, self.neumann[i].get_value(intersected), neumann_vals[j])
        return RayInfo(origin, direction, t, intersected, normals, is_dirichlet, neumann_vals)
    

        
    @dr.syntax
    def get_nearest_distances(self, points, active):
        # active mask needs to be the points where we have nearest neumann boundary!
        # otherwise we need to check the closest point to the boundary, too!
        npoints = dr.width(points)
        min_distance, angles, boundary_points =self.closest_points(points)
        if dr.hint(self.is_full_neumann, mode = 'scalar'):
            return min_distance, boundary_points, dr.full(Bool, False, npoints), dr.inf, Point2f(dr.inf)
        if dr.hint(self.is_full_dirichlet, mode = 'scalar'):
            return min_distance, boundary_points, dr.full(Bool, True, npoints), min_distance, boundary_points
        
        is_dirichlet = self.is_dirichlet_boundary(angles, dr.copy(active))
        if is_dirichlet:
            nearest_dirichlet_dist = Float(min_distance)
            nearest_dirichlet_point = boundary_points
        else:
            nearest_dirichlet_dist = Float(dr.inf)
            nearest_dirichlet_point = Point2f(dr.inf)
        
        i = dr.zeros(UInt, npoints)
        num_dirichlet_arcs = dr.width(self.dirichlet_angles)
        
        while active & (i < num_dirichlet_arcs):
            interval = dr.gather(Point2f, self.dirichlet_angles, i)

            points1 = self.origin + self.radius * \
                Point2f(dr.sin(interval[0]), dr.cos(interval[0]))
            points2 = self.origin + self.radius * \
                Point2f(dr.sin(interval[1]), dr.cos(interval[1]))

            distance1 = dr.norm(points1 - points)
            distance2 = dr.norm(points2 - points)
            dist_mask1 = distance1 < distance2
            nearest_point_temp = dr.select(dist_mask1, points1, points2)
            dist_to_arc = dr.minimum(distance1, distance2)
            dist_mask2 = dist_to_arc < nearest_dirichlet_dist
            nearest_dirichlet_point = Point2f(dr.select(dist_mask2, nearest_point_temp, nearest_dirichlet_point))
            nearest_dirichlet_dist = dr.select(dist_mask2, dist_to_arc, nearest_dirichlet_dist)
            i += 1
        return min_distance, boundary_points, is_dirichlet, nearest_dirichlet_dist, nearest_dirichlet_point
        
        
    @dr.syntax
    def is_dirichlet_boundary(self, angles: Float, active: Bool = Bool(True)):
        num_angles = dr.width(angles)
        if dr.hint(self.is_full_dirichlet, mode = "scalar"):
            return Bool(True)
        elif dr.hint(self.is_full_neumann, mode = 'scalar'):
            return Bool(False)
        else:
            if angles < self.init_angle_partition:
                angles += 2 *dr.pi
            i = dr.zeros(UInt, num_angles)
            is_dirichlet = dr.zeros(Bool, num_angles)
            num_dirichlet = dr.width(self.dirichlet_angles)
            while active & (i < num_dirichlet):
                interval = dr.gather(Point2f, self.dirichlet_angles, i)
                interval_found = (angles >= interval[0]) & (
                    angles < interval[1])
                is_dirichlet |= (interval_found & active)
                active &= ~interval_found
                i += 1
        return is_dirichlet
    
    @dr.syntax
    def boundary_interaction(self, points, 
                             radius_fnc : callable = None, star_generation = True, 
                             max_radius = Float(dr.inf), conf_numbers : list[UInt32] = [UInt32(0)]):
        min_distance, boundary_point, is_dirichlet, nearest_dirichlet_d, nearest_dirichlet_p = self.get_nearest_distances(points, Bool(True))
        
        if dr.hint(self.is_full_neumann, mode = "scalar"):
            merge_dirichlet = Bool(False)
            is_neumann = Bool(True)
            nearest_dirichlet_d = Float(self.radius)
        elif dr.hint(self.is_full_dirichlet, mode = "scalar"):
            merge_dirichlet = Bool(False)
            is_neumann = Bool(False)
        else:
            merge_dirichlet = (nearest_dirichlet_d < (5*dr.epsilon(Float)))
            is_dirichlet |=  merge_dirichlet 
            is_neumann = ~is_dirichlet 
            
        radius = nearest_dirichlet_d if radius_fnc is None else radius_fnc(nearest_dirichlet_d)
        radius = dr.minimum(radius, max_radius)
        boundary_dir = dr.normalize(boundary_point - points)
        # boundary normal defined to the outside of the shape, be careful with this
        boundary_normal = dr.normalize(boundary_point - self.origin) # be careful with inside outside cases
        boundary_normal = -boundary_normal if self.inside else boundary_normal
        is_epsilon_shell = ((min_distance < self.epsilon))
        on_boundary = (is_neumann & (min_distance < 200 * dr.epsilon(Float)))
        
        
        # Set the dirichlet_values to the correct channels.
        num_conf = len(conf_numbers)
        nearest_dirichlet_val = dr.zeros(ArrayXf, shape = (num_conf, dr.width(points)))
        if dr.hint(not self.is_full_neumann, mode = "scalar"):
            if dr.hint(self.num_conf_d == 1, mode = 'scalar'):
                dirichlet = self.dirichlet[0].get_value(boundary_point)
                for i in range(num_conf):
                    nearest_dirichlet_val[i] = dirichlet
            else:
                for i in range(self.num_conf_d): 
                    for j, conf in enumerate(conf_numbers):
                        nearest_dirichlet_val[j] = dr.select(i == conf, 
                                                             self.dirichlet[i].get_value(boundary_point), 
                                                             nearest_dirichlet_val[j])
                        
        is_far = min_distance > self.inf_distance
        curvature = 1/self.radius

        bi = BoundaryInfo(points, on_boundary, radius, min_distance, is_far, boundary_point, curvature, nearest_dirichlet_d,
                          nearest_dirichlet_val, nearest_dirichlet_p, boundary_dir, boundary_normal, is_dirichlet, 
                          is_neumann, is_epsilon_shell, UInt32(0), UInt32(0))
        
        if dr.hint(not self.is_full_dirichlet, mode = 'scalar'):
            if dr.hint(star_generation, mode = 'scalar'):
                bi = self.star_generation(bi)
        return bi
            
    def star_generation(self, bi):
        bi.is_star = bi.is_n & (bi.r > bi.d)
        # If we are very close to the boundary, we will stamp it to the boundary
        diff_centers = bi.origin - self.origin
        diff_dist = dr.norm(diff_centers)
        # only works for  strictly convex boundaries
        angle_centers = correct_angle(dr.atan2(diff_centers[0], diff_centers[1]))
        # cosine theorem
        cos_alpha = (-dr.sqr(bi.r) + dr.sqr(self.radius) + dr.sqr(diff_dist)) / (2 * self.radius * diff_dist)
        angle_diff = dr.acos(cos_alpha)
        angle1 = correct_angle(angle_centers - angle_diff)
        angle2 = correct_angle(angle_centers + angle_diff)
        n1 = - Point2f(dr.sin(angle1), dr.cos(angle1))
        n2 = - Point2f(dr.sin(angle2), dr.cos(angle2))
        bi.x1 = self.origin - self.radius * n1
        bi.x2 = self.origin - self.radius * n2
        vec_c1 = bi.x1 - bi.origin
        vec_c2 = bi.x2 - bi.origin
        bi.angle1 = correct_angle(dr.atan2(vec_c1[0], vec_c1[1]))
        bi.angle2 = correct_angle(dr.atan2(vec_c2[0], vec_c2[1]))
        bi.angle1_adj = dr.copy(bi.angle1)
        bi.angle2_adj = dr.copy(bi.angle2)
        bi.gamma1 = angle1 - bi.angle1
        bi.gamma2 = angle2 - bi.angle2
        return bi
        
    def closest_points(self, points):
        vecs = points - self.origin
        min_distance = dr.norm(vecs) - self.radius
        angles = correct_angle(dr.atan2(vecs[0], vecs[1]))
        boundary_points = self.origin + self.radius * Point2f(dr.sin(angles), dr.cos(angles))
        min_distance *= -1 if self.inside else 1
        return dr.abs(min_distance), angles, boundary_points
    
    def get_closest_dist(self, points):
        min_distance, _, _ =self.closest_points(points)
        return min_distance
    
    def get_distance_correction(self, points):
        return mi.Float(1)

    def inside_closed_surface(self, points : Point2f, L : Float, conf_numbers : list[UInt32] = None):
        vecs = points - self.origin
        return (dr.norm(vecs) <= self.radius), L
    
    def inside_closed_surface_mask(self, points : Point2f):
        vecs = points - self.origin
        return (dr.norm(vecs) <= self.radius)

    def get_opt_params(self, param_dict: dict, opt_params: list):
        #self.dirichlet.get_opt_params(param_dict, opt_params)
        #self.neumann.get_opt_params(param_dict, opt_params)
        pass
                
    def get_opt_params_shape(self, param_dict: dict, opt_params: list):
        for key in opt_params:
            vals = key.split(".")
            boundary_name = vals[0]
            boundary_type = vals[1]
            param = vals[2]
            if (param == "radius") and (boundary_type == "dirichlet") and (boundary_name == self.name):
                param_dict[f"{self.name}.dirichlet.radius"] = self.radius
            elif (param == "origin") and (boundary_type == "dirichlet") and (boundary_name == self.name):
                param_dict[f"{self.name}.dirichlet.origin"] = self.origin
            elif (boundary_name == self.name):
                raise Exception(
                    f"CircleShape ({self.name}) does not have a parameter called \"{param}\"")
        

    def update(self, optimizer):
        #self.dirichlet.update(optimizer)
        #self.neumann.update(optimizer)
        pass      
    def update_shape(self, optimizer):
        for key in optimizer.keys():
            vals = key.split(".")
            name = vals[0]
            type = vals[1]
            param = vals[2]
            if (name == self.name) & (param == "radius") & (type == "dirichlet"):
                self.radius = optimizer[key]
            elif (name == self.name) & (param == "origin") & (type == "dirichlet"):
                self.origin = optimizer[key]
        
    def zero_grad(self):
        self.dirichlet.zero_grad()
        self.neumann.zero_grad()
        
    def zero_grad_shape(self):
        if dr.grad_enabled(self.radius):
            dr.set_grad(self.radius, 0.0)
        if dr.grad_enabled(self.origin):
            dr.set_grad(self.origin, 0)


    def create_result_on_boundary(self, result, film_points, resolution=1024):
        return create_circle_from_result(result, resolution=resolution)

    
    def sketch_image(self, ax, bbox, resolution, channel = 1, image = None, color_factor = 0.8):
        points = create_image_points(bbox, resolution, spp = 1, centered=True)
        result = dr.select(self.inside_closed_surface_mask(points), 1.0, 0.0)
        image_i, tensor = create_image_from_result(result=result, resolution = resolution)
        
        image = np.zeros([resolution[0], resolution[1], 3]) if image is None else image
        image[:,:,channel] += image_i[0] * color_factor
        #image_b = self.get_boundary_image(bbox, resolution)
        #image += image_b
        ax.imshow(image)
        ax.set_axis_off()
        return image
    
    
    def sketch(self, ax, bbox, resolution, colors = ["red", "orange"], fill = False, sketch_center = False, lw = 3):
        origin_s = point2sketch(self.origin, bbox, resolution)
        origin_s = np.array([origin_s[0][0] - 0.5, origin_s[1][0] - 0.5])
        if sketch_center:
            ax.scatter(origin_s[0], origin_s[1], color = colors[0], s = 5)
            return
        radius_x, radius_y, radius = dist2sketch(self.radius, bbox, resolution)
        radius_x = radius_x[0]
        radius_y = radius_y[0]
        angles1 = self.neumann_angles[0].numpy() * 180 / np.pi
        angles2 = self.neumann_angles[1].numpy() * 180 / np.pi
        sphere = patches.Ellipse(origin_s, radius_x * 2, radius_y * 2, linewidth= lw,
                                fill = fill, color = colors[0], label = self.name)
        ax.add_patch(sphere)
        for angle1, angle2 in zip(angles1, angles2):
            neumann_arc = patches.Arc(origin_s,  2 * radius_x, 2 * radius_y, angle = -90, theta1=angle1, theta2=angle2, linewidth= lw, color=colors[1])
            ax.add_patch(neumann_arc)  
        

    
    def create_boundary_points(self, distance: float, res: int, spp: int, discrete_points : bool = True):
        with dr.suspend_grad():
            distance = -distance if self.inside else distance
            points = create_circle_points(
                self.origin, self.radius + distance, res, spp, discrete_points=discrete_points)
            normal_dir = dr.normalize(points - self.origin)
        return points, points, normal_dir

    def create_boundary_result(self, result, points = None, resolution = 256):
        with dr.suspend_grad():
            tensor, tensor_mi = create_circle_from_result(result, resolution)
        return tensor, tensor_mi
    
    def create_boundary_coefficient(self, tensor_mi, name = "boundary-val"):
        def boundary_val(points, parameters):
            resolution = dr.width(parameters["bval"].array)
            vec = points - self.origin
            angles = correct_angle(dr.atan2(vec[0], vec[1]))
            angles = correct_angle(angles)
            angles = dr.select(angles<0, angles + 2 * dr.pi, angles)
            indices = angles / (2 * dr.pi) * resolution
            index0 =  UInt32(dr.floor(indices)) % resolution
            index1 = (index0 + 1) % resolution
            residual = indices - Float(index0)
            vals0 = dr.gather(Float, parameters["bval"].array, index0)
            vals1 = dr.gather(Float, parameters["bval"].array, index1)
            return vals0 * (1-residual) + vals1 * residual

        coeffs = []
        for i in range(tensor_mi.shape[0]):
            parameters = {}
            parameters["bval"] = Float(tensor_mi[i])
            coeffs.append(FunctionCoefficient(name, dict(parameters), boundary_val))
        return coeffs
    
    def set_normal_derivative(self, tensor_mi):
        self.normal_derivatives = self.create_boundary_coefficient(tensor_mi, "normal-derivative")
        return self.normal_derivatives
    
    def jakobian_to_boundary(self, bi : BoundaryInfo, distance : Float):
        return 1 + distance / self.radius
    
    def get_normal_derivative(self, points : Point2f):
        num_conf = len(self.normal_derivatives)
        normal_ders = dr.zeros(ArrayXf, shape = [num_conf, dr.width(points)])
        for i in range(num_conf):
            normal_ders[i] = self.normal_derivatives[i].get_value(points)
        return normal_ders
    
    def get_max_intersection_dist(self, bi : BoundaryInfo):
        return 2 * self.radius
    
    def generate_sdf_grid(self, resolution, box_length = 2, box_center = [0,0],
                          wrapping = "clamp", interpolation = "cubic", redistance = True, high_res = 2048):
        bbox = [[box_center[0] - box_length/2, box_center[1] - box_length/2],
                [box_center[0] + box_length/2, box_center[1] + box_length/2]]
        points = create_image_points(bbox = bbox, resolution = resolution, spp = 1, centered = True)
        inside = self.inside_closed_surface_mask(points)
        min_d, a, b = self.closest_points(points)
        distance = dr.select(inside, -min_d, min_d)
        image_np, image_mi = create_image_from_result(distance, resolution)
        return SDFGrid(image_np[0], box_length, box_center, self.dirichlet, self.epsilon, self.inf_distance,
                       self.inside, self.name,
                       normal_derivative_dist= self.normal_derivative_dist, wrapping = wrapping, interpolation=interpolation, 
                       redistance = redistance, high_res = high_res, low_res=resolution[0])
    

        
    def move_circle_fd(self, fd_step, type = "x"):
        origin1 = [self.origin[0], self.origin[1]]
        origin2 = [self.origin[0], self.origin[1]]
        radius1 = self.radius[0]
        radius2 = self.radius[0]
        if type == "x":
            origin1 = [origin1[0] + fd_step / 2, origin1[1]]
            origin2 = [origin2[0] - fd_step / 2, origin2[1]]
        elif type == "y":
            origin1 = [origin1[0], origin1[1] + fd_step / 2]
            origin2 = [origin2[0], origin2[1] - fd_step / 2]
        elif type == "r":
            radius1 = radius1 + fd_step/2
            radius2 = radius2 - fd_step/2
        else: 
            raise Exception("There is no such type.")
        
        circle1 =  CircleShape(origin1, radius1, dirichlet = self.dirichlet, neumann = self.neumann, epsilon = self.epsilon)
        circle2 =  CircleShape(origin2, radius2, dirichlet = self.dirichlet, neumann = self.neumann, epsilon = self.epsilon)
        return circle1, circle2
        
