import drjit as dr
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from ..Coefficient import *
from ..utils.helpers import *
from .interaction import *
from mitsuba import Float, Point2f, Bool
from enum import IntEnum

class NEE(IntEnum):
    Normal = 0,
    Special = 1,
    BruteForce = 2

class Shape(object):
    def __init__(self, is_full_dirichlet = False, is_full_neumann = False, single_closed_shape = True, 
                 epsilon=1e-5, inside = True, derivative_dist = 1e-2, inf_distance = 10):
        self.is_full_dirichlet = is_full_dirichlet
        self.is_full_neumann = is_full_neumann
        self.single_closed = single_closed_shape
        self.epsilon = epsilon
        self.min_star_radius = self.epsilon * 2 # Check the paper for better min distance!
        self.name = "boundary"
        self.inside = inside  
        self.normal_derivative_dist = derivative_dist
        self.measured_current = False # If the Boundary condition given as currents or normal derivatives.
        self.has_continuous_neumann = True
        self.has_delta = False
        self.NEE = NEE.Normal
        self.inf_distance = inf_distance
    
    
    def star_generation(self, bi: BoundaryInfo) -> BoundaryInfo:
        return bi
    
    def inside_closed_surface(self, points, L, conf_numbers):
        return Bool(False), dr.zeros(ArrayXf, shape = (len(conf_numbers), dr.width(points)))
    
    def inside_closed_surface_mask(self, L):
        return Bool(False)
    
    def ray_intersect(self, bi : BoundaryInfo, direction : Point2f, conf_numbers : list[UInt32] = None) -> RayInfo:
        pass
    
    def boundary_interaction(points : Point2f, 
                             radius_fnc : callable = None, max_radius = None, 
                             star_generation = True, conf_numbers : list[UInt32] = [UInt32[0]]) -> BoundaryInfo:
        pass
    
    def get_opt_params_shape(self, param_dict: dict, opt_params: list):
        pass
    
    def update_shape(self, optimizer):
        pass
    
    def zero_grad_shape(self):
        pass
    
    def sketch(self,ax, bbox, resolution, colors = ['red'], fill = False):
        pass

    def sketch_image(self,ax, bbox, resolution, channel= 0, colors = ["red"], image = None, color_factor = 0.6):
        pass
    
    def create_boundary_points(self, distance : float, resolution : int, spp : int):
        pass

    def create_neumann_points(self, resolution : int, spp : int):
        pass

    def create_boundary_result(self, result, resolution):
        pass

    def create_boundary_coefficient(self, tensor_mi, name = "boundary-val"):
        pass

    def set_normal_derivative(self, tensor_mi, name = "normal-derivative"):
        pass
    
    def normal_derivative_from_result(self, result : Float, film_points : Point2f, resolution : int):
        pass
    
    def create_normal_der_coefficient(self, tensor_mi):
        pass
    
    def jakobian_to_boundary(self, bi : BoundaryInfo, distance : Float):
        pass
    
    def get_max_intersection_dist(self, bi : BoundaryInfo):
        pass
    
    def get_distance_correction(self, points):
        pass
    def get_point_neumann(self, bi : BoundaryInfo, conf_number : UInt32) -> tuple[list[Float], list[Float], list[Float], list[Point2f]]:
        return [], [], [], []
    
    def sampleNEE(self, bi : BoundaryInfo, sample : Float, conf_number : UInt32) -> tuple[Float, Float, Float, Point2f]:
        return Float(0), Float(0), Float(0), Point2f(0)
    
    def create_volume_points(self, resolution = [256, 256], spp = 1):
        points = create_image_points(self.bbox, resolution, spp, centered = True)
        active = self.inside_closed_surface_mask(points)
        indices = dr.compress(active)
        points = dr.gather(type(points), points, indices)
        return points
    
    def accum_tput(self, tput : Float, bi : BoundaryInfo):
        pass