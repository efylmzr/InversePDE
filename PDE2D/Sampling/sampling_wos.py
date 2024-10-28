import mitsuba as mi 
from mitsuba import Float, Bool
from .special import *
from ..utils import *
from ..utils.helpers import *


def sample_cosk_direction(sample, direction, k : Float = Float(2)): #k>2
    # sample symmetrically w.r.t. the direction with cos(theta / k)
    right_sphere = sample >= 0.5
    sample = dr.select(right_sphere, 2 * sample - 1, 2 * sample)
    angle_shift = k * dr.asin(sample * dr.sin(dr.pi /k))
    angle_shift = dr.select(right_sphere, angle_shift, -angle_shift)
    angle_initial = dr.atan2(direction[1], direction[0])
    angle = angle_initial + angle_shift
    sampled_direction = mi.Point2f(dr.cos(angle), dr.sin(angle))
    pdf = dr.rcp(2 * k * dr.sin(dr.pi /k)) * dr.cos(angle_shift / k)
    return sampled_direction, pdf

def pdf_cosk_direction(sampled_direction, direction, k): #k>2
    angle_diff = dr.abs(dr.acos(dr.dot(sampled_direction, direction)))
    return dr.rcp(2 * k * dr.sin(dr.pi /k)) * dr.cos(angle_diff / k)

@dr.syntax
def sample_star_direction(sample,  half_space_mask : Bool, boundary_normal : mi.Point2f) -> tuple[mi.Point2f, mi.Float]:
    angle = mi.Float(0)
    direction = mi.Point2f(0)
    pdf = mi.Float(0)
    if half_space_mask:
        angle = mi.Float((sample - 0.5) * dr.pi)
        direction = mi.Point2f(dr.sin(angle), dr.cos(angle))
        direction = dr.normalize(to_world_direction(direction, boundary_normal))
        pdf = Float(1/dr.pi)
    else:
        angle = mi.Float(2 * dr.pi * sample)
        direction = mi.Point2f(dr.sin(angle), dr.cos(angle))
        pdf = Float(1 / (2 * dr.pi))
    return direction, pdf

def sample_uniform_direction(sample):
    theta = 2 * dr.pi * sample
    return mi.Point2f(dr.sin(theta), dr.cos(theta)), Float(1/(2 * dr.pi))

def sample_uniform_boundary(sample, origin, radius):
    direction, pdf = sample_uniform_direction(sample)
    sampled_points = origin + radius * direction
    return sampled_points, pdf / radius

def sample_cosine_direction(sample : Float, direction : mi.Vector2f) -> tuple[mi.Vector2f, Float, Float]:
    upper_sphere = sample >= 0.5
    sample = dr.select(upper_sphere, 2 * sample - 1, 2 * sample)
    angle_shift = dr.asin(2 * sample - 1)
    abs_dot_prod = dr.sqrt(1 - dr.square(2 * sample -1))
    angle_initial = dr.atan2(direction[1], direction[0])
    angle = angle_initial + angle_shift
    sampled_direction = mi.Point2f(dr.cos(angle), dr.sin(angle))
    sign = dr.select(upper_sphere, Float(1), Float(-1))
    sampled_direction *= sign
    return sampled_direction, abs_dot_prod / 4, sign


def sample_cosine_boundary(sample : Float, origin : mi.Point2f, radius : Float, direction : mi.Vector2f) -> tuple[mi.Vector2f, Float, Float]:
    sampled_direction, pdf, sign = sample_cosine_direction(sample, direction)
    point = origin + radius * sampled_direction
    return point, pdf / radius, sign

def sample_cosine_boundary_antithetic(sample, origin, radius, direction, active):
    angle_shift = dr.asin(2 * sample - 1)
    direction1 = mi.Vector2f(dr.sin(angle_shift), dr.cos(angle_shift))
    direction2 = mi.Vector2f(dr.sin(angle_shift), -dr.cos(angle_shift))
    direction1 = to_world_direction(direction1, direction)
    direction2 = to_world_direction(direction2, direction)
    point1 = mi.Point2f(dr.select(active, origin + radius * direction1, origin))
    point2 = mi.Point2f(dr.select(active, origin + radius * direction2, origin))
    return point1, point2, dr.cos(angle_shift) / (2 * radius) , 2
    

def pdf_cosine_boundary_(sampled_direction, R, direction):
    return 1/4 * dr.abs(dr.dot(dr.normalize(sampled_direction), dr.normalize(direction))) / R

def pdf_cosine_boundary(points, origin, R, direction):
    d = dr.normalize(points - origin)
    return pdf_cosine_boundary_(d, R, direction)

def sample_uniform_volume(sample, origin, radius):
    r =  radius * dr.sqrt(sample[0])
    theta = 2 * dr.pi * sample[1]
    return mi.Vector2f(origin +  r * mi.Vector2f(dr.cos(theta),dr.sin(theta))), dr.rcp(dr.pi * dr.sqr(radius))



def sample_sec_direction(sample : Float, direction : mi.Vector2f, threshold : Float = Float(0.49 * dr.pi)):
    negative = sample >= 0.5
    sample = dr.select(negative, 2 * sample - 1, 2 * sample)
    angle_shift = sample_sec_angle(sample, threshold)
    angle_shift *= dr.select(negative, -1., 1)

    angle_initial = dr.atan2(direction[1], direction[0])
    angle = angle_initial + angle_shift
    sampled_direction = mi.Vector2f(dr.cos(angle), dr.sin(angle))
    return sampled_direction

@dr.syntax
def pdf_sec_direction(dir : mi.Vector2f, direction : mi.Vector2f, threshold : Float = Float(0.49 * dr.pi)):
    pdf = Float(0)
    sec = dr.rcp(dr.dot(dir, direction))
    csc_d = dr.rcp(dr.sin(threshold))
    sec_d = dr.rcp(dr.cos(threshold))
    normalization = 0.5 * dr.log((1 + csc_d)/(-1 + csc_d)) + (dr.pi/2 - threshold) * sec_d

    if (sec > 0) & (sec < sec_d):
        pdf = sec
    elif (sec >= sec_d):
        pdf = sec_d
    return pdf / normalization * 0.5


@dr.syntax
def sample_sec_angle(sample : Float, threshold : Float = Float(0.49 * dr.pi)):
    csc_d = dr.rcp(dr.sin(threshold))
    sec_d = dr.rcp(dr.cos(threshold))

    th_val = 0.5 * dr.log((1 + csc_d)/(-1 + csc_d))
    normalization = th_val + (dr.pi/2 - threshold) * sec_d
    sample *= normalization
    
    sampled_p = Float(0)
    if sample < th_val:
        exp = dr.exp(2 * sample)
        sampled_p = dr.asin((exp - 1)/(exp + 1))
    else:
        sampled_p = threshold + (sample - th_val) / (normalization - th_val) * (dr.pi / 2 - threshold)
    return sampled_p


@dr.syntax
def pdf_sec_angle(angle : Float, threshold : Float = Float(0.49 * dr.pi)): # pdf with respect to secant.
    pdf = Float(0)
    sec = dr.rcp(dr.cos(angle))
    csc_d = dr.rcp(dr.sin(threshold))
    sec_d = dr.rcp(dr.cos(threshold))
    normalization = 0.5 * dr.log((1 + csc_d)/(-1 + csc_d)) + (dr.pi/2 - threshold) * sec_d
    if (angle >= 0) & (angle < threshold):
        pdf = sec
    elif (angle >= threshold) & (angle <= dr.pi/2):
        pdf = sec_d
    return pdf / normalization

@dr.syntax
def eval_dP_norm(radius : Float, σ : Float) -> Float: 
    # used in directional derivative
    sqrtσ = dr.sqrt(σ)
    z = radius * sqrtσ
    result = Float(0)
    if z < 0.001:
        result = dr.rcp(dr.pi * dr.square(radius))
    else:
        result = sqrtσ * dr.rcp(2 * dr.pi * radius * i1(radius * sqrtσ))
    return result


def eval_Pσr_(r, R, sigma, in_mask = Bool(False)): # multiplied with 2 * pi * r version
    z = R * dr.sqrt(sigma)
    y  = r / R
    return dr.select(in_mask, Qσ(y, z), eval_Pσrs_(R, sigma))

def eval_Pσrs_(R, sigma):
    return dr.rcp(i0(R * dr.sqrt(sigma)))