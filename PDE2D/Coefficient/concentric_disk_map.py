import drjit as dr
import mitsuba as mi
# Peter Shirley and Kenneth Chiu in 1997 derived an area preserving map between square to disc.

@dr.syntax
def square_to_disk(point : mi.Point2f):
    if dr.abs(point[0]) > dr.abs(point[1]):
        u = point[0] * dr.cos(dr.pi/4 * point[1] / point[0])
        v = point[0] * dr.sin(dr.pi/4 * point[1] / point[0])
    else:
        u = point[1] * dr.sin(dr.pi/4 * point[0] / point[1])
        v = point[1] * dr.cos(dr.pi/4 * point[0] / point[1])
    return mi.Point2f(u, v)

@dr.syntax
def disk_to_square(point : mi.Point2f):
    r = dr.norm(point)
    if dr.abs(point[0]) >= dr.abs(point[1]):
        p =  r * mi.Point2f(dr.sign(point[0]), 4 / dr.pi * dr.atan2(point[1], dr.abs(point[0])))
    else:
        p = r * mi.Point2f(4 / dr.pi * dr.atan2(point[0], (dr.abs(point[1]) + dr.epsilon(mi.Float))), dr.sign(point[1]))
    return p

@dr.syntax
def jakobian(point: mi.Point2f):
    
    if dr.abs(point[0]) > dr.abs(point[1]):
        A = dr.pi * point[1] / (4 * point[0])
        cos_A = dr.cos(A)
        sin_A = dr.sin(A)
        mat = mi.Matrix2f(cos_A + A * sin_A, -dr.pi/4 * sin_A,
                        sin_A - A * cos_A, dr.pi/4 * cos_A)
    else:
        B = dr.pi * point[0] / (4 * point[1]) 
        cos_B = dr.cos(dr.pi / 4 + B)
        sin_B = dr.sin(dr.pi / 4 + B)
        mat = mi.Matrix2f(dr.pi/4 * cos_B, sin_B - B * cos_B,
                       -dr.pi/4 * sin_B, cos_B + B * sin_B)
    return mat