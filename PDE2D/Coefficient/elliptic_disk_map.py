import mitsuba as mi
import drjit as dr
# Peter Shirley and Kenneth Chiu in 1997 derived an area preserving map between square to disc.

def square_to_disk(point : mi.Point2f, radius : mi.Float = 1, origin : mi.Point2f = mi.Point2f(0)):
    x = (point[0] - origin[0]) / radius
    y = (point[1] - origin[1]) / radius
    u = x * dr.sqrt(1 - dr.sqr(y) / 2)
    v = y * dr.sqrt(1 - dr.sqr(x) / 2)
    return origin + radius * mi.Point2f(u,v)

def disk_to_square(point: mi.Point2f, radius: mi.Float = 1, origin: mi.Point2f = mi.Point2f(0)):
    u = (point[0] - origin[0]) / radius
    v = (point[1] - origin[1]) / radius
    u2 = dr.sqr(u)
    v2 = dr.sqr(v)
    x = 0.5 * (dr.sqrt(2 + u2 - v2 + 2 * dr.sqrt(2) * u) - 
               dr.sqrt(2 + u2 - v2 - 2 * dr.sqrt(2) * u))
    y = 0.5 * (dr.sqrt(2 - u2 + v2 + 2 * dr.sqrt(2) * v) -
               dr.sqrt(2 - u2 + v2 - 2 * dr.sqrt(2) * v))
    return origin + radius * mi.Point2f(x, y)

def jakobian(point: mi.Point2f, radius : mi.Float = 1, origin : mi.Point2f = mi.Point2f(0)):
    x = (point[0] - origin[0]) / radius
    y = (point[1] - origin[1]) / radius
    x2 = dr.sqr(x)
    y2 = dr.sqr(y)
    a11 = dr.sqrt(1 - y2 / 2) # du/dx
    a12 = -x * y / dr.sqrt(4 - 2 * y2) # du/dy
    a21 = -x * y / dr.sqrt(4 - 2 * x2) # dv/dx
    a22 = dr.sqrt(1 - x2 / 2) # dv/dy
    return mi.Matrix2f(a11, a12, a21, a22)

def inverse_jakobian(point : mi.Point2f, radius : mi.Float = 1, origin : mi.Point2f = mi.Point2f(0)):
    u = (point[0] - origin[0]) / radius
    v = (point[1] - origin[1]) / radius
    u2 = dr.sqr(u)
    v2 = dr.sqr(v)
    c11 = 1/(2 * dr.sqrt(2 + u2 - v2 + 2*dr.sqrt(2) * u))
    c12 = 1/(2 * dr.sqrt(2 + u2 - v2 - 2*dr.sqrt(2) * u))
    c21 = 1/(2 * dr.sqrt(2 - u2 + v2 + 2*dr.sqrt(2) * v))
    c22 = 1/(2 * dr.sqrt(2 - u2 + v2 - 2*dr.sqrt(2) * v))
    
    a11 = 1/2 * c11 * (2 * u + 2*dr.sqrt(2)) - 1/2 * c12 * (2 * u - 2*dr.sqrt(2)) # dx/du
    a12 = -c11 * v + c12 * v  # dx/dv
    a21 = -c21 * u + c22 * u # dy/du
    a22 = 1/2 * c21 * (2 * v + 2*dr.sqrt(2)) - 1/2 * c22 * (2 * v - 2*dr.sqrt(2)) # dy/dv
    
    c11_u = -2 * dr.sqr(c11) * c11 * (2 * u + 2*dr.sqrt(2))
    c12_u = -2 * dr.sqr(c12) * c12 * (2 * u - 2*dr.sqrt(2))
    c21_u =  4 * dr.sqr(c21) * c21 * u
    c22_u =  4 * dr.sqr(c22) * c22 * u
    
    c11_v = 4 * dr.sqr(c11) * c11 * v
    c12_v = 4 * dr.sqr(c12) * c12 * v
    c21_v = -2 * dr.sqr(c21) * c21 * (2 * v + 2*dr.sqrt(2))
    c22_v = -2 * dr.sqr(c22) * c22 * (2 * v - 2*dr.sqrt(2))
    
    a11_u = 1/2 * c11_u * (2 * u + 2*dr.sqrt(2)) + c11 - 1/2 * c12_u * (2 * u - 2 * dr.sqrt(2)) - c12 # d2x/du2
    a12_v = -c11 + c12 + v * (c12_v - c11_v)# d2x/dv2
    a21_u = -c21 + c22 + u * (c22_u - c21_u) # d2y/du2
    a22_v = 1/2 * c21_v * (2 * v + 2*dr.sqrt(2)) + c21 - 1/2 * c22_v * (2 * v - 2*dr.sqrt(2)) - c22  # d2y/dv2
    
    return mi.Matrix2f(a11, a12, a21, a22), mi.Matrix2f(a11_u, a12_v, a21_u, a22_v)
