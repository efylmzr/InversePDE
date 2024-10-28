import drjit as dr
from ..utils.helpers import *
from ..utils.sketch import *
from matplotlib import patches as patches
from mitsuba import Point2f, Vector2f, Bool, Float, UInt32 
from PDE2D import ArrayXf
from ..Sampling import sample_star_direction, sample_sec_direction, pdf_sec_direction


class BoundaryInfo:
    DRJIT_STRUCT = {
        'origin' : Point2f,
        'on_boundary': Bool,
        'r' : Float,
        'd' : Float,
        'is_far' : Bool,
        'bpoint' : Point2f,
        'dd' : Float,
        'dval' : ArrayXf,
        'dpoint' : Point2f,
        'bdir' : Vector2f,
        'bn' : Vector2f,
        'is_d' : Bool,
        'is_n' : Bool,
        'is_e' : Bool,
        'shape' : UInt32, # Index of the closest boundary
        'd_shape' : UInt32, # Index of the closest dirichlet boundary
        'is_star' : Bool,
        'x1' : Point2f,
        'x2' : Point2f,
        'angle1' : Float,
        'angle2' : Float,
        'angle1_adj': Float,
        'angle2_adj' : Float,
        'gamma1' : Float,
        'gamma2' : Float,
    }
    def __init__(self, origin = None, on_boundary = None, r = None, d = None, is_far = None, 
                 bpoint = None, curvature = None,
                dd = None, dval = None, dpoint = None, bdir = None, bn = None,
                is_d = None, is_n = None, is_e = None, shape = None, d_shape = None, 
                is_star = None, x1 = None, x2 = None, angle1 = None, angle2 = None, 
                angle1_adj = None, angle2_adj = None, gamma1 = None, gamma2 = None):
        self.origin = origin
        self.on_boundary = on_boundary
        self.r = r
        self.d = d
        self.is_far = is_far 
        self.bpoint = bpoint
        self.curvature = curvature
        self.dd = dd
        self.dval = dval
        self.dpoint = dpoint
        self.bdir = bdir
        self.bn = bn
        self.is_d = is_d
        self.is_n = is_n
        self.is_e = is_e
        self.shape = shape
        self.d_shape = d_shape
        self.is_star = is_star
        self.x1 = x1
        self.x2 = x2
        self.angle1 = angle1
        self.angle2 = angle2
        self.angle1_adj = angle1_adj
        self.angle2_adj = angle2_adj
        self.gamma1 = gamma1
        self.gamma2 = gamma2
    
    def sample_recursive(self, sample : Float): # samples the full star
        direction, pdf = sample_star_direction(sample, self.on_boundary & self.is_star, self.bn)
        sphere_points =  self.origin + self.r * direction
        return direction, sphere_points, pdf
    
    def pdf_recursive(self):
        return dr.select(self.on_boundary & self.is_star, 1/dr.pi, 1/(2 * dr.pi))
    
    @dr.syntax
    def sample_brute_force(self, sample : Float, mis_rate : Float = Float(0.5), threshold : Float = Float(0.49 * dr.pi)):
        "Applies a bit more sophisticated sampling scheme, mis between uniform and secant weighted distribution if we are near the boundary."
        sampled_dir = Vector2f(0)
        pdf = Float(0)
        if self.on_boundary | (self.d < (dr.abs(dr.rcp(self.curvature)) / 10)):
            direction = dr.select(self.on_boundary, self.bn, self.bdir)
            sec_mask = sample < mis_rate
            sample = dr.select(sec_mask, sample / mis_rate, (sample - mis_rate) / (1.0-mis_rate))
            if sec_mask:
                sampled_dir = sample_sec_direction(sample, direction, threshold)
            else:
                sampled_dir, _, _ = self.sample_recursive(sample)
            
            pdf_sec = pdf_sec_direction(sampled_dir, direction, threshold)
            pdf_rec = self.pdf_recursive()
            pdf = mis_rate * pdf_sec + (1.0-mis_rate) * pdf_rec
        else:
            sampled_dir, _, pdf  = self.sample_recursive(sample)
        
        return sampled_dir, pdf
        

    def sample_neumann(self, sample : Float, on_boundary : Bool): # samples only the boundary part
        # inside case
        angle_diff = correct_angle(self.angle2_adj - self.angle1_adj)
        angle_in = correct_angle(self.angle1_adj + angle_diff * sample)
        direction_in = Vector2f(dr.sin(angle_in), dr.cos(angle_in))
        # on-boundary case
        angle_n = correct_angle(dr.atan2(self.bn[0], self.bn[1]))
        angle_n1 = dr.pi/2 - correct_angle(self.angle1_adj - angle_n)
        angle_n2 = dr.pi/2 - correct_angle(angle_n - self.angle2_adj)
        angle_sum = angle_n1 + angle_n2
        angle_diff_b = dr.pi - angle_sum
        angle_boundary = sample * angle_sum 
        angle_boundary += dr.select(angle_boundary > angle_n2, angle_diff_b, 0)
        angle_boundary -= dr.pi/2
        dir_n = Vector2f(dr.sin(angle_boundary), dr.cos(angle_boundary))
        direction_boundary = to_world_direction(dir_n, self.bn)
        direction = dr.select(on_boundary, direction_boundary, direction_in)
        pdf = dr.select(on_boundary, 1/angle_sum, 1/(angle_diff))
        return direction, pdf
    
    def pdf_neumann():
        pass
    
    def update_angles(self, angle1, angle2): 
        # this is done if we do not want to sample the whole Neumann part of the star
        self.angle1_adj = angle1
        self.angle2_adj = angle2
    
    def sketch_stars(self, ax, indices, bbox, resolution, color_star = "green", color_critical = "blue"):
        actives = self.is_star.numpy()[indices]
        origins = point2sketch(self.origin, bbox, resolution).numpy()[:, indices]
        radii_x, radii_y, radii = dist2sketch(self.r, bbox, resolution)
        radii_x = radii_x.numpy()[indices]
        radii_y = radii_y.numpy()[indices]
        x1s = point2sketch(self.x1, bbox, resolution).numpy()[:, indices]
        x2s = point2sketch(self.x2, bbox, resolution).numpy()[:, indices]
        angles1 = self.angle1.numpy()[indices] * 180 / np.pi
        angles2 = self.angle2.numpy()[indices] * 180 / np.pi
        for origin, radius_x, radius_y, x1, x2, angle1, angle2, active \
            in zip(origins, radii_x, radii_y, x1s, x2s, angles1, angles2, actives):
                if active:
                    #star = patches.Ellipse(origin, 
                    #                radius_x * 2, radius_y * 2, 
                    #                fill = False, color = color_star)
                    star = patches.Arc(origin,  2 * radius_x, 2 * radius_y, angle = -90, 
                                              theta1=angle2, theta2=angle1, linewidth=2.5, color=color_star)
                    center = patches.Ellipse(origin, 4, 4, 
                                    fill = True, color = color_star)
                    critical1 = patches.Ellipse([x1[0], x1[1]], 4, 4, 
                                        fill = True, color = color_critical)
                    critical2 = patches.Ellipse([x2[0], x2[1]], 4, 4, 
                                        fill = True, color = color_critical)
                    #ax.arrow(x1[0], x1[1], n1[0], n1[1], color = color_critical,
                    #    edgecolor = "none", width = 0.03)
                    #ax.arrow(x2[0], x2[1], n2[0], n2[1], color = color_critical,
                    #    edgecolor = "none", width = 0.03)
                    ax.add_patch(star)
                    ax.add_patch(center)
                    ax.add_patch(critical1)
                    ax.add_patch(critical2)


class RayInfo:
    DRJIT_STRUCT = {
        'origin' : Point2f,
        'direction' : Vector2f,
        't' : Float,
        'intersected' : Point2f,
        'normal' : Vector2f,
        'is_dirichlet' : Bool,
        'neumann' : ArrayXf # We want to get multiple neumann values at once.
    }
    def __init__(self, origin = None, direction = None, t = None, intersected = None, normal = None, is_dirichlet = None,  neumann = None):
        self.origin = origin
        self.direction = direction
        self.t = t
        self.intersected = intersected
        self.normal = normal
        self.is_dirichlet = is_dirichlet
        self.neumann = neumann 

        


    
