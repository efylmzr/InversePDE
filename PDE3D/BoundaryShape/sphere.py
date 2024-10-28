from PDE3D.BoundaryShape.boundary_shape import BoundaryInfo
from .boundary_shape import *

class Sphere(Shape):
    DRJIT_STRUCT = {
        'origin' : mi.Point3f,
        'radius' : mi.Float
        }
    def __init__(self, origin=[0, 0, 0], radius=1,  epsilon=1e-5, inf_distance = 10, dirichlet : list[Coefficient] = []):
        super().__init__(True, epsilon, inf_distance, dirichlet)
        self.origin = mi.Point3f(origin)
        self.radius = mi.Float(radius)
        dr.make_opaque(self.origin)
        dr.make_opaque(self.radius)

        #self.shape_mi = mi.load_dict({
        #                       'type': 'sphere',
        #                       'center': origin,
        #                       'radius': radius,
        #                       'bsdf': {
        #                       'type': 'diffuse'}})
        
        self.scene = mi.load_dict({
                            "type" : "scene",
                            "sphere": {
                               'type': 'sphere',
                               'center': origin,
                               'radius': radius,
                               'bsdf': {
                               'type': 'diffuse'}}})

    def inside_closed_surface(self, points : mi.Point3f, L : mi.Float, conf_numbers : list[mi.UInt32] = None):
        vecs = points - self.origin
        return (dr.norm(vecs) <= self.radius), L
    
    def inside_closed_surface_mask(self, points : mi.Point3f):
        vecs = points - self.origin
        return (dr.norm(vecs) <= self.radius)

    def boundary_interaction(self, points: mi.Point3f, conf_numbers : list[mi.UInt32] = [mi.UInt32(0)]) -> BoundaryInfo:
        dir_point = points - self.origin
        norm_dir = dr.norm(dir_point)
        dist_boundary = dr.abs(self.radius - norm_dir)
        boundary_point = self.origin + dir_point / norm_dir * self.radius
        
        num_conf = len(conf_numbers)
        dirichlet = dr.zeros(ArrayXf, shape = (num_conf, dr.width(points)))

        if dr.hint(self.num_conf_d == 1, mode = 'scalar'):
            dirichletval = self.dirichlet[0].get_value(boundary_point)
            for i in range(num_conf):
                dirichlet[i] = dirichletval
        else:
            for i in range(self.num_conf_d): 
                for j, conf in enumerate(conf_numbers):
                    dirichlet[j] = dr.select(i == conf, 
                                            self.dirichlet[i].get_value(boundary_point), 
                                            dirichlet[j])
        return BoundaryInfo(points, dist_boundary, mi.Float(dist_boundary), dist_boundary < self.epsilon,  mi.Point3f(boundary_point), dirichlet)

    
    