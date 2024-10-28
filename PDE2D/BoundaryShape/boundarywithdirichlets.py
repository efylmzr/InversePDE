from PDE2D.BoundaryShape.interaction import BoundaryInfo
from PDE2D.BoundaryShape import CircleShape
from .boundary_shape import *
from ..utils.helpers import *
from ..utils.sketch import *
from .interaction import *
from scipy.spatial import Voronoi

from ..utils.imageUtils import create_circle_from_result, create_circle_points
from ..Coefficient import *
class BoundaryWithDirichlets(Shape):
    def __init__(self, out_boundary : Shape,
                 dirichlet_boundaries : list[Shape] = [], dirichlet_values : list[list] = None,
                 epsilon = 1e-5, name : str = "DirichletShapes"):
        #assert len(dirichlet_values) == len(dirichlet_boundaries)
        
        self.single_shape_closed = True
        self.name = name
        self.out_boundary = out_boundary
        self.in_boundaries = dirichlet_boundaries
        self.out_boundary.inside = True
        self.is_full_dirichlet = self.out_boundary.is_full_dirichlet
        self.max_distance = self.out_boundary.max_distance
        self.single_closed = True
        self.neumann = self.out_boundary.neumann # Only neumann is the out boundary
        # If the shape has some NEE structure.
        self.NEE = self.out_boundary.NEE
        self.has_continuous_neumann = self.out_boundary.has_continuous_neumann
        self.has_delta = self.out_boundary.has_delta
        # If the boundary condition given as currents or normal derivatives.
        self.measured_current = self.out_boundary.measured_current 
        self.epsilon = epsilon
        self.out_boundary.epsilon = epsilon
        self.bbox = self.out_boundary.bbox
        self.bbox_length = max(self.bbox[1][1] - self.bbox[0][1], self.bbox[1][0] - self.bbox[0][0])
        self.update_in_boundaries(dirichlet_boundaries, dirichlet_values)

    def update_in_boundaries(self, in_boundaries : list[Shape], dirichlet_values : list[list] = None):
        self.num_shapes = len(in_boundaries)
        if dirichlet_values is None:
            dirichlet_values = [[0] for i in range(self.num_shapes)]
        
        #if self.num_shapes > 0:
        #    self.num_conf_d = len(dirichlets[0])
        #else:
        #    self.num_conf_d = 1

        self.in_boundaries = in_boundaries
        self.is_full_neumann = (self.num_shapes == 0) & self.out_boundary.is_full_neumann
        for  i, d_shape in enumerate(self.in_boundaries):
            assert d_shape.is_full_dirichlet
            #assert len(dirichlet_values[i]) == self.num_conf_d
            d_shape.inside = False
            d_shape.epsilon = self.epsilon

        self.update_in_boundary_dirichlets(dirichlet_values)

    
    def get_origins(self):
        origins = []
        for d_shape in self.in_boundaries:
            origins.append(d_shape.origin.numpy().squeeze())
        return np.array(origins)

    def update_in_boundaries_circle(self, origins : list, radius : float = 0.01, dirichlet_values : list[list] = None):
        in_boundaries = []
        for origin in origins:
            in_boundaries.append(CircleShape(origin = origin, radius = radius))
        self.update_in_boundaries(in_boundaries, dirichlet_values)
    
    @dr.syntax
    def update_in_boundary_dirichlets(self, dirichlet_values : list[list]):
        self.dirichlets = []
        for i in range(self.num_shapes):
            self.dirichlets.append(ArrayXf(dirichlet_values[i]))
        dr.make_opaque(self.dirichlets)

    @dr.syntax
    def ray_intersect(self, bi : BoundaryInfo, direction : Point2f, conf_numbers : list [UInt32] = None):  # change this for outside!
        ri = self.out_boundary.ray_intersect(bi, direction, conf_numbers)
        for boundary in self.in_boundaries:
            ri_in = boundary.ray_intersect(bi, direction, conf_numbers)
            if (ri_in.t < ri.t) & (ri_in.t >= 0):
                ri.t = Float(ri_in.t)
                ri.intersected = Point2f(ri_in.intersected)
                ri.normal = Point2f(ri_in.normal)
                ri.is_dirichlet = Bool(ri_in.is_dirichlet)
                if dr.hint(conf_numbers is not None):
                    num_conf = len(conf_numbers)
                    if dr.hint(num_conf == 1):
                        ri.neumann[0] = Float(0)
                    else:
                        for j in range(num_conf):
                            ri.neumann[j] = Float(0)
        return ri
    
    @dr.syntax
    def boundary_interaction(self, points, 
                             radius_fnc : callable = None, star_generation = True, 
                             max_radius = Float(dr.inf), conf_numbers : list[UInt32] = [UInt32(0)]):
        num_conf = len(conf_numbers)
        assert len(self.dirichlets) == self.num_shapes
        if self.num_shapes>0:
            assert (num_conf == dr.width(self.dirichlets[0])) or (dr.width(self.dirichlets[0]) == 1)
        bi = self.out_boundary.boundary_interaction(points, radius_fnc, star_generation = False, conf_numbers = conf_numbers)
        for (i,boundary) in enumerate(self.in_boundaries):
            bi_in = boundary.boundary_interaction(points, radius_fnc, star_generation = False, conf_numbers = conf_numbers)
            bi_in.shape = UInt32(i+1)
            bi_in.d_shape = UInt32(i+1)
            bi_in.dval = ArrayXf(self.dirichlets[i])
            bi = merge_boundary_info(bi, bi_in)
        
        bi.r = dr.minimum(max_radius, bi.r)
        
        if dr.hint(not self.is_full_dirichlet, mode = 'scalar'):
            if dr.hint(star_generation, mode = 'scalar'):
                bi = self.star_generation(bi)
        return bi
            
    def star_generation(self, bi):
        return self.out_boundary.star_generation(bi)

    
    def sampleNEE(self, bi : BoundaryInfo, sample : Float, conf_number : UInt32) -> tuple[Float, Float, Float, Point2f]:
        return self.out_boundary.sampleNEE(bi, sample, conf_number)
    
    def get_point_neumann(self, bi : BoundaryInfo, conf_number : UInt32) -> tuple[list[Float], list[Float], list[Float], list[Point2f]]:
        return self.out_boundary.get_point_neumann(bi, conf_number)

    @dr.syntax
    def inside_closed_surface(self, points : Point2f, L : ArrayXf, conf_numbers = list[UInt32]):
        num_conf = len(conf_numbers)
        if self.num_shapes>0:
            assert (num_conf == dr.width(self.dirichlets[0])) or (dr.width(self.dirichlets[0]) == 1)
        
        active, L = self.out_boundary.inside_closed_surface(points, L)
        for i, boundary in enumerate(self.in_boundaries):
            mask, L = boundary.inside_closed_surface(points, L)
            active &= ~mask
            if mask:
                L += ArrayXf(self.dirichlets[i])
        return active, L
    
    @dr.syntax
    def inside_closed_surface_mask(self, points : Point2f):
        active = self.out_boundary.inside_closed_surface_mask(points)
        for i, boundary in enumerate(self.in_boundaries):
            mask = boundary.inside_closed_surface_mask(points)
            active &= ~mask
        return active
                
    def get_opt_params_shape(self, param_dict: dict, opt_params: list):
        self.out_boundary.get_opt_params_shape(param_dict, opt_params)
        for boundary in self.in_boundaries:
            boundary.get_opt_params_shape(param_dict, opt_params)
                
    def update_shape(self, optimizer):
        #if post_process is not None:
        #    post_process(optimizer)
        #self.out_boundary.update_shape(optimizer)
        #for boundary in self.in_boundaries:
        #    boundary.update_shape(optimizer)
        self.in_boundaries[0].update_shape(optimizer)

    def zero_grad_shape(self):
        self.out_boundary.zero_grad_shape()
        for boundary in self.in_boundaries:
            boundary.zero_grad_shape()

    
    def sketch_image(self, ax, bbox, resolution, channel = 0, colors = ["orange", "green"], image = None, 
                     color_factor = 0.6):        
        image = np.zeros([resolution[0], resolution[1], 3])
        for shape in self.in_boundaries:
            image = shape.sketch_image(ax, bbox, resolution, channel = channel, 
                                       image =image, color_factor=color_factor)
        ax.imshow(image)

        self.out_boundary.sketch(ax, bbox, resolution, colors = colors)
        return image
            
    def sketch(self, ax, bbox, resolution, colors = ["orange","green", "red"], fill = False, sketch_center = False, sketch_in_boundaries = True):
        if sketch_in_boundaries:
            for shape in self.in_boundaries:
                shape.sketch(ax, bbox, resolution, colors = [colors[2], colors[0]], fill = fill, sketch_center = sketch_center)
        self.out_boundary.sketch(ax, bbox, resolution, colors = colors[0:2])

    def create_boundary_points(self, distance: float, res: int, spp: int, discrete_points : bool = True):
        with dr.suspend_grad():
            points = []
            normals = []
            s_points = []
            p, s_p, n = self.out_boundary.create_boundary_points(distance, res, spp, discrete_points)
            points.append(p)
            normals.append(n)
            s_points.append(s_p)
            for boundary in self.in_boundaries:
                p, s_p, n = boundary.create_boundary_points(distance, res, spp, discrete_points)
                points.append(p)
                s_points.append(s_p)
                normals.append(n)
        return points, s_p, normals

    def create_boundary_result(self, result, resolution):
        with dr.suspend_grad():
            tensor = []
            tensor_mi = []
            t, t_mi = self.out_boundary.create_boundary_result(result[0], resolution)
            tensor.append(t)
            tensor_mi.append(t_mi)
            for i, boundary in enumerate(self.in_boundaries):
                t, t_mi = boundary.create_boundary_result(result[i+1], resolution)
                tensor.append(t)
                tensor_mi.append(t_mi)
        return tensor, tensor_mi

    def create_boundary_coefficient(self, tensor_mi, name = "boundary-val"):
        with dr.suspend_grad():
            self.boundary_coeff = []
            self.boundary_coeff.append(self.out_boundary.create_boundary_coefficient(tensor_mi[0], f'{name}-0'))
            for i, boundary in enumerate(self.in_boundaries):
                self.boundary_coeff.append(boundary.create_boundary_coefficient(tensor_mi[i+1], f'{name}-{i}'))
        return self.boundary_coeff

    
    #def set_normal_derivative(self, tensor_mi, name = "normal-derivative"):
    #    self.normal_derivative = []
    #    self.normal_derivative.append(self.out_boundary.set_normal_derivative(tensor_mi[0], f"{name}-0"))
    #    for i, boundary in enumerate(self.in_boundaries):
    #            self.normal_derivative.append(boundary.set_normal_derivative(tensor_mi[i+1], f'{name}-{i}'))
    #    return self.normal_derivative
    
    #@dr.syntax
    #def jakobian_to_boundary(self, bi : BoundaryInfo, distance = None):
        # This one computes the minimum of the distances to compute the boundary result.
    #    jak = Float(0)
    #    for i, shape in enumerate(self.in_boundaries):
    #        if bi.d_shape == UInt32(i+1):
    #            jak = shape.jakobian_to_boundary(bi, max_distance)
    #    return jak
    
    
    def get_normal_derivative(self, points : Point2f):
        return self.in_boundaries[0].get_normal_derivative(points)
    
    def get_jacobian_factor(self, bi : BoundaryInfo, distance : float):
        ri = self.ray_intersect(bi, bi.bn)
        distance = dr.minimum(distance, 0.3 * ri.t)
        return self.in_boundaries[0].jakobian_to_boundary(bi, distance)

    
    def get_opt_params(self, param_dict: dict, opt_params: list):
        self.in_boundaries[0].get_opt_params_shape(param_dict, opt_params)
    
    def update(self, optimizer):
        self.out_boundary.update(optimizer)
        for shape in self.in_boundaries:
            shape.update(optimizer)
        
    def zero_grad(self):
        self.out_boundary.zero_grad()
        for shape in self.in_boundaries:
            shape.zero_grad()

def merge_boundary_info(bi1, bi2, radius_fnc : callable = None):
    # merges 2 boundary informations created with different shapes.
    # it does not handle the star generation!
    # Here bi2 must be only dirichlet, it is created for circlewithdirichletsshape
    mask = bi1.d < bi2.d
    mask_d = bi1.dd < bi2.dd
    bi = BoundaryInfo()
    bi.origin = Point2f(bi1.origin)
    #bi.on_boundary = dr.select(mask, bi1.on_boundary, bi2.on_boundary)
    bi.on_boundary = bi1.on_boundary
    bi.d = dr.select(mask, bi1.d, bi2.d)
    bi.is_far = dr.select(mask, bi1.is_far, bi2.is_far)
    bi.dd = dr.select(mask_d, bi1.dd, bi2.dd)
    bi.r = bi.dd
    bi.bpoint = dr.select(mask, bi1.bpoint, bi2.bpoint)
    bi.curvature = dr.select(mask, bi1.curvature, bi2.curvature)
    bi.dval = dr.select(mask_d, bi1.dval, bi2.dval)
    bi.dpoint = dr.select(mask_d, bi1.dpoint, bi2.dpoint)
    bi.bdir = dr.select(mask, bi1.bdir, bi2.bdir)
    bi.bn = dr.select(mask, bi1.bn, bi2.bn)
    bi.is_d = dr.select(mask, bi1.is_d, bi2.is_d)
    bi.is_n = dr.select(mask, bi1.is_n, bi2.is_n)
    bi.is_e = dr.select(mask, bi1.is_e, bi2.is_e)
    bi.shape = dr.select(mask, bi1.shape, bi2.shape)
    bi.d_shape = dr.select(mask_d, bi1.d_shape, bi2.d_shape)
    return bi 