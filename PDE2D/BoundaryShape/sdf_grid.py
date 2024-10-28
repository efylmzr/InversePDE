
from .boundary_shape import *
from ..utils.helpers import *
from ..utils.sketch import *
from ..Coefficient import ConstantCoefficient
import skfmm
from scipy import optimize
from .interaction import *

class SDFGrid(Shape):
    def __init__(self, tensor_np = np.zeros([16, 16]), box_length = 2.1, box_center = [0,0], dirichlet : list[Coefficient] = [], 
                 epsilon=1e-5, inf_distance=10, inside = False, name = "boundary", type = "sdf", 
                 normal_derivative_dist = 0.01, wrapping = "clamp", interpolation = "cubic", redistance = True,
                 translation = [0,0], low_res = None, high_res = 2048):
        super().__init__(True, single_closed_shape=True, epsilon=epsilon,
                         inf_distance=inf_distance, inside = inside)
        low_res = tensor_np.shape[0] if low_res is None else low_res
        tensor_np = tensor_np.squeeze()
        if not (tensor_np.shape[0] == tensor_np.shape[1]):
            raise Exception("You need to specify a square image.")
        self.name = name
        self.res =  low_res
        self.resolution = [low_res, low_res]
        self.high_res_factor = int(high_res / self.res)
        self.res_high =  high_res
        self.type = type
        self.box_center = mi.Point2f(box_center)
        self.box_length = mi.Float(box_length)
        self.bbox = [[box_center[0] - box_length/2, box_center[1] - box_length/2],
                     [box_center[0] + box_length/2, box_center[1] + box_length/2]]
        
        
        #self.dx = box_length / self.res
        self.dx_high = box_length / self.res_high
        #self.threshold = self.dx * 0.25
        # SDF shape only supports dirichlet boundary conditions.
        self.is_full_dirichlet = True
        # Dirichlet Boundary Values
        self.dirichlet = dirichlet
        self.normal_derivative_dist = normal_derivative_dist
        self.redistance = redistance
        self.translation_x = mi.Float(translation[0])
        self.translation_y = mi.Float(translation[1])
        self.tensor = mi.TensorXf(tensor_np[..., np.newaxis])

        dr.make_opaque(self.translation_x)
        dr.make_opaque(self.translation_y)
        dr.make_opaque(self.tensor)
        self.wrapping = wrapping
        self.interpolation = interpolation

        if redistance:
            self.tensor = self.redistance_tensor(self.tensor)
        self.texture = self.update_texture(self.tensor)
        dr.make_opaque(self.texture)

        self.num_conf_d = len(dirichlet)
        self.num_conf_n = 1
        assert (self.num_conf_d == self.num_conf_n) or (self.num_conf_n == 1) or (self.num_conf_d == 1)
    
    
    def redist(self):
        self.tensor = self.redistance_tensor(self.tensor)
        self.texture = self.update_texture(self.tensor)
        
    def update_texture(self, tensor):
        # Creating the texture!
        wrap_mode = None
        if self.wrapping == "clamp":
            wrap_mode = dr.WrapMode.Clamp
        elif self.wrapping == "mirror":
            wrap_mode = dr.WrapMode.Mirror
        elif self.wrapping == "repeat":
            wrap_mode = dr.WrapMode.Repeat
        else:
            raise Exception("Such wrapping is not defined.")

        texture = mi.Texture2f(
            tensor, wrap_mode=wrap_mode, use_accel=False, migrate=False, filter_mode=dr.FilterMode.Linear)
        return texture
    
    def rasterize_tensor(self, res, texture):
        resolution = [res, res]
        points = create_image_points(self.bbox, resolution, spp = 1, centered = True)
        vals = self.get_texture_value(points, texture)   
        image, _ = create_image_from_result(vals,  resolution)
        return image[0]
    
    def get_texture_value(self, points: mi.Point2f, texture):
        points_bbox = self.get_position(points)
        if (self.interpolation == "cubic"):
            texture_val =  texture.eval_cubic(points_bbox)[0]
        elif (self.interpolation == "linear"):
            texture_val = texture.eval(points_bbox)[0]
        else:
            raise Exception(
                f"There is no interpolation called \"{self.interpolation}\"")
        return texture_val
    
    def redistance_tensor(self, tensor):
        texture_low = self.update_texture(tensor)
        high_array  = self.rasterize_tensor(self.res_high, texture_low)
        high_array = skfmm.distance(high_array.astype(np.float64), dx = self.dx_high)

        tensor_high = mi.TensorXf(high_array[..., np.newaxis])
        texture_high = self.update_texture(tensor_high)
        low_array = self.rasterize_tensor(self.res, texture_high)
        tensor_low = mi.TensorXf(low_array[..., np.newaxis])
        return tensor_low
    
    def update(self, optimizer : mi.ad.Optimizer):
        for key in optimizer.keys():
            vals = key.split(".")
            name = vals[0]
            type = vals[1]
            param = vals[2]
            if (name == self.name) & (param == "tensor") & (type == "dirichlet"):
                # Apply redistancing.
                optimizer[key] = self.redistance_tensor(optimizer[key])
                self.tensor = optimizer[key]
                dr.make_opaque(self.tensor)
                self.texture = self.update_texture(self.tensor)
            
            if (name == self.name) & (param == "translation_x") & (type == "dirichlet"):
                self.translation_x = optimizer[key]
                dr.make_opaque(self.translation_x)
            
            if (name == self.name) & (param == "translation_y") & (type == "dirichlet"):
                self.translation_y = optimizer[key]
                dr.make_opaque(self.translation_y)
            
    def get_opt_params(self, param_dict: dict, opt_params: list):
        self.dirichlet.get_opt_params(param_dict, opt_params)

    def get_opt_params_shape(self, param_dict: dict, opt_params: list):
        for key in opt_params:
            vals = key.split(".")
            boundary_name = vals[0]
            boundary_type = vals[1]
            param = vals[2]
            if (param == "tensor") and (boundary_type == "dirichlet") and (boundary_name == self.name):
                param_dict[f"{self.name}.dirichlet.tensor"] = self.tensor
            elif (param == "translation_x") and (boundary_type == "dirichlet") and (boundary_name == self.name):
                param_dict[f"{self.name}.dirichlet.translation_x"] = self.translation_x
            elif (param == "translation_y") and (boundary_type == "dirichlet") and (boundary_name == self.name):
                param_dict[f"{self.name}.dirichlet.translation_y"] = self.translation_y
            elif (boundary_name == self.name):
                raise Exception(
                    f"SDFGrid ({self.name}) does not have a parameter called \"{param}\"")
             
    def get_position(self, points_ : mi.Point2f):
        points = points_ - mi.Point2f(self.translation_x, self.translation_y)
        "Get the new positions of the points normalized for the bbox."
        x = (points[0] - self.bbox[0][0]) / (self.bbox[1][0] - self.bbox[0][0])
        y = 1.0 - (points[1] - self.bbox[0][1]) / (self.bbox[1][1] - self.bbox[0][1])
        return mi.Point2f(x, y) 
    
    
    def get_closest_dist(self, points : mi.Point2f):
        dist = self.get_texture_value(points, self.texture)
        return dist
    
    @dr.syntax
    def ray_intersect(self, bi : BoundaryInfo, direction, on_boundary : mi.Bool, max_step = 100):
        with dr.suspend_grad():
            dist = self.get_closest_dist(bi.origin)
            close_mask = dist < 100 * dr.epsilon(mi.Float)
            normal = self.get_normal(bi.origin)
            point = dr.select(close_mask & (dr.dot(normal, direction) > 0),
                           bi.origin + normal * 200 * dr.epsilon(mi.Float),
                           bi.origin)
            active = mi.Bool(True)
            i = mi.UInt(0)
            while (active & (i < max_step)):
                i += 1
                dist = self.get_closest_dist(point)
                close_mask = dist < 10 * dr.epsilon(mi.Float)
                far_mask = dist > self.inf_distance
                active &= (~far_mask & ~close_mask)
                point = dr.select(active, point + direction * mi.Point2f(dist, dist), point)
                point = dr.select(far_mask, dr.inf, point) 
            point = dr.select(active, dr.inf, point)
            t = dr.norm(point - bi.origin)
            normals = self.get_normal(point)
        
        return RayInfo(bi.origin, direction, t, point, normals, mi.Bool(True), mi.Float(0))
    
    def get_distance_correction(self, points):
        grad = self.get_grad(points)
        return dr.norm(grad)
    
    def get_grad_hessian(self, points : mi.Point2f):
        # We are not using high res texture here as we only compute
        # gradient at the boundary
        dilate_x = self.bbox[1][0] - self.bbox[0][0]
        dilate_y = self.bbox[1][1] - self.bbox[0][1]
        points_bbox = self.get_position(points)
        eval_result = self.texture.eval_cubic_hessian(points_bbox)
        grad = eval_result[1][0] / mi.Point2f(dilate_x, -dilate_y)
        hessian_ = eval_result[2][0]
        hessian_x = hessian_[0, 0] / (dilate_x ** 2)
        hessian_y = hessian_[1, 1] / (dilate_y ** 2)
        hessian_xy = hessian_[0, 1] / (-dilate_x * dilate_y)
        hessian = mi.Matrix2f([[hessian_x, hessian_xy],
                               [hessian_xy, hessian_y]])
        return grad, hessian
    
                
    def get_grad(self, points: mi.Point2f):
        dilate_x = self.bbox[1][0] - self.bbox[0][0]
        dilate_y = self.bbox[1][1] - self.bbox[0][1]
        points_bbox = self.get_position(points)
        eval_result = self.texture.eval_cubic_grad(points_bbox)
        grad = eval_result[1][0] / mi.Point2f(dilate_x, -dilate_y)
        return mi.Point2f(grad)
    
                
    def get_normal(self, points : mi.Point2f):
        grad = self.get_grad(points)
        return dr.normalize(grad)
    
    def get_boundary_distance(self, points : mi.Point2f):
        dist0 = self.get_texture_value(points, self.texture)
        grad0 = self.get_grad(points)
        norm0 = dr.norm(grad0)
        normal_dir = dr.detach(grad0 / norm0)
        
        # We first take a step in the sphere direction. 
        # Now we are most probably in the problematic region where norm of 
        # the gradient is not 1.
        points1 = dr.detach(points - normal_dir * dist0)
        dist1 = self.get_texture_value(points1, self.texture)
        grad1 = self.get_grad(points1)
        norm1 = dr.norm(grad1)
        normal1 = dr.detach(grad1 / norm1)
        
        # Now we will take another step.
        points2 = dr.detach(points1 - normal1 * dist1)
        dist2 = self.get_texture_value(points2, self.texture)
        grad2 = self.get_grad(points2)
        norm2 = dr.norm(grad2)
        
        # Now we linearly approximate the norm of the gradient in the direction of the normal dir.
        # We set the loc of points2 to 0 and points1 to dist1.
        x2 = mi.Float(0)
        x1 = mi.Float(dist1)
        # The norm of the gradient along the line will be ax + b near the boundary.
        a = (norm1 - norm2) / (x1 - x2)
        b = norm2
        
        # Now the texture value near the boundary in the normal direction is the integral of this.
        # axˆ2/2 + bx + dist2
        # Zero crossing value (the root) of this function is the following.
        x_zero =  (-b  + dr.sqrt((dr.sqr(b) - 2 * a * dist2))) / a
        return dist0 + dist1 - x_zero
        
        # Now this value is wrong

    
    def boundary_interaction(self, points: mi.Point2f, radius_fnc : callable = None, 
                             star_generation = False, max_radius = Float(dr.inf), conf_numbers : list[mi.UInt32] = [mi.UInt32(0)]) -> BoundaryInfo:

        min_distance = self.get_closest_dist(points)
        bpoints = mi.Point2f(points)
        radius = mi.Float(min_distance)
        if radius_fnc is not None:
            radius = radius_fnc(radius)
        radius = dr.minimum(radius, max_radius)
        is_epsilon_shell = (min_distance < self.epsilon) 
        boundary_normal = self.get_normal(bpoints)
        
        num_conf = len(conf_numbers)
        dirichlet = dr.zeros(ArrayXf, shape = (num_conf, dr.width(points)))

        if dr.hint(self.num_conf_d == 1, mode = 'scalar'):
            dirichletval = self.dirichlet[0].get_value(bpoints)
            for i in range(num_conf):
                dirichlet[i] = dirichletval
        else:
            for i in range(self.num_conf_d): 
                for j, conf in enumerate(conf_numbers):
                    dirichlet[j] = dr.select(i == conf, 
                                            self.dirichlet[i].get_value(bpoints), 
                                            dirichlet[j])

        curvature =  self.compute_curvature(points)
        return BoundaryInfo(points, mi.Bool(False), radius, min_distance, min_distance > self.inf_distance, 
                     bpoints, curvature, min_distance, dirichlet, bpoints, -boundary_normal, boundary_normal, 
                     mi.Bool(True), mi.Bool(False), is_epsilon_shell, UInt32(0), UInt32(0)) 
        
                
    @dr.syntax
    def get_touch_point(self, points, num_steps = 16, active_ = mi.Bool(True)):
        with dr.suspend_grad():
            active = Bool(active_)
            touch_point = mi.Point2f(points)
            d = mi.Float(dr.inf)
            i = mi.UInt(0)
            while (active & (i < num_steps)):
                i+=1
                d = mi.Float(self.get_closest_dist(touch_point))
                active &= dr.abs(d) > dr.epsilon(mi.Float) * 10
                normal = self.get_normal(touch_point)
                touch_point = mi.Point2f(dr.select(active, touch_point - normal * mi.Point2f(d,d) * 0.99, touch_point))
            return touch_point, d
            
    
    def compute_curvature(self, points):
        grad, hessian = self.get_grad_hessian(points)
        norm_grad = dr.norm(grad)
        grad_ = mi.Point2f(-grad[1], grad[0])
        return -(grad_ @ hessian @ grad_) / (norm_grad * dr.sqr(norm_grad))
    
        
    def get_boundary_indices(self, resolution): 
        x_length = self.bbox[1][0] - self.bbox[0][0]
        y_length = self.bbox[1][1] - self.bbox[0][1]
        dx = x_length / resolution[0]
        dy = y_length / resolution[1]
        x, y = dr.meshgrid(dr.arange(mi.Float, resolution[0]), 
                       dr.arange(mi.Float, resolution[1]), indexing='xy')
        
        film_points = mi.Point2f(x, y)
        p = mi.Point2f(self.bbox[0][0], self.bbox[1][1]) +  film_points / mi.Point2f(resolution) *  mi.Point2f(x_length, -y_length)
        p_x = p + mi.Point2f(dx, 0)
        p_y = p + mi.Point2f(0, -dy)
        p_xy = p + mi.Point2f(dx, -dy)
        
        p_pos = self.get_closest_dist(p) > 0
        p_x_pos = self.get_closest_dist(p_x) > 0
        p_y_pos = self.get_closest_dist(p_y) > 0
        p_xy_pos = self.get_closest_dist(p_xy) > 0

        non_boundary_mask = (p_pos & p_x_pos & p_y_pos & p_xy_pos) | (~p_pos & ~p_x_pos & ~p_y_pos & ~p_xy_pos)
        boundary_mask =  ~non_boundary_mask 
        boundary_mask_np = boundary_mask.numpy()
        film_points_np = film_points.numpy().astype(np.int16).T
        return mi.Point2f((film_points_np[boundary_mask_np].T).astype(np.float32))
    
    def create_boundary_points(self, distance: float, res: int, spp: int, discrete_points : bool = True, seed : int = 42):
        with dr.suspend_grad():
            resolution = [res, res]
            film_points = self.get_boundary_indices(resolution)
            film_points = dr.repeat(film_points, spp) + mi.Point2f(0.5, 0.5)
            
            if not discrete_points:
                sampler = mi.load_dict({'type': 'independent'})
                sampler.seed(seed, dr.width(film_points))
                film_points_ = film_points + sampler.next_2d() - 1/2
            else:
                film_points_ = film_points
    
            
            points_ = (mi.Point2f(self.bbox[0][0], self.bbox[1][1]) +  
                      film_points_ / mi.Point2f(resolution) *  
                      (mi.Point2f(self.bbox[1][0], self.bbox[0][1]) - mi.Point2f(self.bbox[0][0], self.bbox[1][1])))

            boundary_points, d = self.get_touch_point(points_)
            normal_dir = self.get_normal(boundary_points)
            points = boundary_points + mi.Point2f(distance, distance) * normal_dir
        return points, points_,  normal_dir

    def create_boundary_result(self, result, points = None, res = 256):
        if points is None:
            raise Exception("Specify the points corresponding to the estimates in result.")
        with dr.suspend_grad():
            resolution = [res, res]
            i2, i1 = get_position_bbox(points, self.bbox)
            i2 = dr.minimum(mi.UInt(resolution[1] * i2), resolution[1] - 1)
            i1 = dr.minimum(mi.UInt(resolution[0] * i1), resolution[0] - 1)

            dim = 1 if result.ndim == 1 else result.shape[0]
            tensor = dr.zeros(mi.TensorXf, shape = (dim, resolution[0], resolution[1]))
            index = dr.zeros(mi.TensorXu, shape = (resolution[0], resolution[1]))
            n = resolution[0] * resolution[1]
            dr.scatter_add(index.array, mi.UInt(1), i1 * resolution[1] + i2)
            #if dim == 1:
            #    dr.scatter_add(tensor.array, i1 * resolution[1] + i2, result)
            for i in range(dim):
                dr.scatter_add(tensor.array, result[i], i * n  +  i1 * resolution[1] + i2)
        
        index = dr.select(index.array == 0, 1, index.array)
        index = dr.reshape(mi.TensorXu, index, shape = (resolution[0], resolution[1]))
        tensor /= index
        return tensor.numpy(), tensor

    def create_boundary_coefficient(self, tensor_mi, name = "boundary-val"):
        coeffs = []
        for i in range(tensor_mi.shape[0]):
            coeffs.append(TextureCoefficient(name, self.bbox, tensor_mi[i].numpy().squeeze(), interpolation = "nearest"))
        return coeffs
    
    def set_normal_derivative(self, tensor_mi, name = "normal-derivative"):
        self.normal_derivatives = self.create_boundary_coefficient(tensor_mi, name = "normal-derivative")
        return self.normal_derivatives
    
    def get_normal_derivative(self, points : Point2f):
        #points_, _ = self.get_touch_point(points)
        num_conf = len(self.normal_derivatives)
        normal_ders = dr.zeros(ArrayXf, shape = [num_conf, dr.width(points)])
        for i in range(num_conf):
            normal_ders[i] = self.normal_derivatives[i].get_value(points)
        return normal_ders
    
    def jakobian_to_boundary(self, bi : BoundaryInfo, distance = None, max_distance : mi.Float = dr.inf):
        distance = self.normal_derivative_dist if distance is None else distance
        distance = dr.minimum(distance, max_distance)
        comp_points = dr.detach(bi.bpoint + bi.bn * distance)
        distance = self.get_closest_dist(comp_points)
        curvature = self.compute_curvature(dr.detach(bi.bpoint))
        return 1 - distance * curvature
    
    def sketch_image(self, ax, bbox, resolution, image = None, channel = 1, color_factor = 0.6):
        points = create_image_points(bbox, resolution, spp = 1, centered=True)
        result = dr.select(self.get_closest_dist(points) < 0, 1.0, 0.0)
        image_i, tensor = create_image_from_result(result=result, resolution = resolution)
        
        image = np.zeros([resolution[0], resolution[1], 3]) if image is None else image
        image[:,:,channel] = image_i * 0.6
        #image_b = self.get_boundary_image(bbox, resolution)
        #image += image_b
        ax.imshow(image * color_factor)
        ax.set_axis_off()
        return image
    
    def get_boundary_image(self, bbox, resolution, channel = 0):
        indices = self.get_boundary_indices(bbox, resolution)
        image = np.zeros([resolution[0], resolution[1], 3])
        image[indices[:,1], indices[:,0], channel] += 1
        #ax.imshow(image)
        return image 
    
    def inside_closed_surface(self, points, L):
        return self.get_closest_dist(points) < 0, L
    
    def inside_closed_surface_mask(self, points):
        return self.get_closest_dist(points) < 0
    
    
    def get_boundary_polyline(self, start = [0,0], step = 0.01):
        points = []
        point, d = self.get_touch_point(mi.Point2f(start))
        points.append(point.numpy())
        first = True
        while True:
            normal = self.get_normal(mi.Point2f(points[-1]))
            tangent = mi.Point2f(normal[1], -normal[0])
            if first:
                tangent_first = mi.Point2f(tangent)
            next_point = mi.Point2f(points[-1]) + tangent * step
            next_point, d = self.get_touch_point(next_point)
            if (not first) and (dr.norm(next_point - point)[0] < 1.1 * step) and dr.dot(dr.normalize(next_point - point), tangent_first)[0] < 0.1:
                break
            else:
                points.append(next_point.numpy())
            first = False
        points = np.array(points).squeeze()
        self.polyline = mi.Point2f(points.T)
        return points
        
    def sketch_boundary_polyline(self, ax, bbox, resolution, esize = 0.2):
        points = self.polyline.numpy().T
        sketch_points = []
        xscale = bbox[1][0] - bbox[0][0]
        yscale = bbox[1][1] - bbox[0][1]
        pointsx = (points[:, 0]-bbox[0][0])/xscale*resolution[1]
        pointsy = resolution[0] - (points[:, 1]-bbox[0][1])/yscale*resolution[0]
        pointsx = np.append(pointsx, pointsx[0])
        pointsy = np.append(pointsy, pointsy[0])
        sketch_points = np.vstack([pointsx, pointsy])
        ax.plot(sketch_points[0,:], sketch_points[1,:], zorder = 0)
        ax.scatter(sketch_points[0,0], sketch_points[1,0], color = "red", zorder = 1, s = 0.1)
        return np.array(sketch_points).T

    
    def vis_images(self, bbox, resolution = [1024, 1024], spp = 16):
        points = create_image_points(bbox, resolution = resolution, spp = spp, centered = False)
        d = self.get_closest_dist(points)
        crossing = dr.select(d<0, 1.0, 0.0)
        grad = self.get_grad(points)
        norm_grad = dr.norm(grad)
        
        d_im, _ = create_image_from_result(d, resolution)
        crossing_im, _ = create_image_from_result(crossing, resolution)
        gradx_im, _ = create_image_from_result(grad[0], resolution)
        grady_im, _ = create_image_from_result(grad[1], resolution)
        normgrad_im, _ = create_image_from_result(norm_grad, resolution)
        return d_im[0], crossing_im[0], gradx_im[0], grady_im[0], normgrad_im[0]
    
    
        